use burn::{
    backend::DispatchDevice,
    tensor::{DType, Tensor as BurnTensor},
};
use burn_backend::{Shape, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    CubeBackend, CubeDevice,
    kernel::untile,
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_cubecl::cubecl::{self, calculate_cube_count_elemwise, prelude::*};

use crate::{
    KernelDescriptor, KernelError, KernelOperation, KernelTarget, PagedAttentionInput,
    PagedAttentionKernel,
};

/// One-launch CubeCL implementation of single-sequence paged decode attention.
///
/// Each logical K/V page remains a distinct device buffer. CubeCL specializes
/// the generated program for the number of page bindings and traverses those
/// bindings directly; no sequence-sized K/V tensor is materialized.
#[derive(Clone, Copy, Debug)]
pub struct CubePagedAttention {
    target: KernelTarget,
}

impl CubePagedAttention {
    /// Creates a CubeCL kernel for a WGPU or CUDA execution target.
    pub fn new(target: KernelTarget) -> Result<Self, KernelError> {
        match target {
            KernelTarget::Wgpu | KernelTarget::Cuda => Ok(Self { target }),
            _ => Err(KernelError::new(
                "CubeCL paged attention currently supports WGPU or CUDA targets",
            )),
        }
    }

    /// Reports whether this implementation can consume the validated input.
    ///
    /// The first correctness slice accepts `f32` inference tensors only. The
    /// portable kernel remains the fallback for other dtypes or devices.
    pub fn supports(&self, input: &PagedAttentionInput) -> bool {
        if input.dtype() != DType::F32 {
            return false;
        }
        let device = input.query().device();
        match (self.target, device.as_dispatch()) {
            (KernelTarget::Wgpu, DispatchDevice::Cube(CubeDevice::Wgpu(_)))
            | (KernelTarget::Cuda, DispatchDevice::Cube(CubeDevice::Cuda(_))) => true,
            _ => false,
        }
    }
}

impl PagedAttentionKernel for CubePagedAttention {
    fn descriptor(&self) -> KernelDescriptor {
        let name = match self.target {
            KernelTarget::Wgpu => "cubecl-wgpu-paged-attention",
            KernelTarget::Cuda => "cubecl-cuda-paged-attention",
            _ => unreachable!("constructor rejects non-CubeCL targets"),
        };
        KernelDescriptor::new(name, KernelOperation::PagedAttentionDecode, self.target)
    }

    fn execute(&self, input: PagedAttentionInput) -> Result<BurnTensor<4>, KernelError> {
        if !self.supports(&input) {
            return Err(KernelError::new(format!(
                "{} does not support dtype {:?} on device {:?}",
                self.descriptor().name(),
                input.dtype(),
                input.query().device()
            )));
        }

        let (query, pages, config) = input.into_parts();
        let query = cube_primitive(query, "query")?;
        let mut keys = Vec::with_capacity(pages.len());
        let mut values = Vec::with_capacity(pages.len());
        for (index, page) in pages.into_iter().enumerate() {
            let (key, value) = page.into_parts();
            keys.push(cube_primitive(key, &format!("key page {index}"))?);
            values.push(cube_primitive(value, &format!("value page {index}"))?);
        }

        let output = launch_paged_attention(query, keys, values, config.groups());
        Ok(BurnTensor::<4>::from_primitive::<CubeBackend>(output))
    }
}

fn cube_primitive(tensor: BurnTensor<4>, name: &str) -> Result<CubeTensor, KernelError> {
    tensor
        .try_into_primitive::<CubeBackend>()
        .map(untile)
        .map_err(|error| {
            KernelError::new(format!(
                "{name} is not a non-autodiff CubeCL tensor: {error:?}"
            ))
        })
}

fn launch_paged_attention(
    query: CubeTensor,
    keys: Vec<CubeTensor>,
    values: Vec<CubeTensor>,
    groups: usize,
) -> CubeTensor {
    debug_assert_eq!(keys.len(), values.len());
    let [_, query_heads, _, head_dim]: [usize; 4] = query
        .meta
        .shape()
        .dims()
        .try_into()
        .expect("validated query has rank four");
    let output = empty_device_dtype(
        query.client.clone(),
        query.device.clone(),
        Shape::new([1, query_heads, 1, head_dim]),
        query.dtype,
    );
    let client = query.client.clone();
    let dtype = query.dtype;
    let work_items = query_heads * head_dim;
    let cube_dim = CubeDim::new(&client, work_items);
    let cube_count = calculate_cube_count_elemwise(&client, work_items, cube_dim);
    let key_args = keys.into_iter().map(CubeTensor::into_tensor_arg).collect();
    let value_args = values
        .into_iter()
        .map(CubeTensor::into_tensor_arg)
        .collect();

    paged_attention::launch(
        &client,
        cube_count,
        cube_dim,
        query.into_tensor_arg(),
        key_args,
        value_args,
        output.clone().into_tensor_arg(),
        groups,
        1.0_f32 / (head_dim as f32).sqrt(),
        dtype_to_storage_type(dtype),
    );

    output
}

/// Straightforward correctness kernel: one work item owns one output element
/// and performs a stable online softmax across every populated page token.
/// Recomputing each token score per output element is deliberate in this first
/// slice; later kernels can add cooperative reductions without changing the
/// public operator contract.
#[cube(launch)]
fn paged_attention<F: Float>(
    query: &Tensor<F>,
    keys: &Sequence<Tensor<F>>,
    values: &Sequence<Tensor<F>>,
    output: &mut Tensor<F>,
    groups: usize,
    scale: f32,
    #[define(F)] _dtype: ElemType,
) {
    let head_dim = query.shape(3);
    let output_elements = query.shape(1) * head_dim;
    if ABSOLUTE_POS >= output_elements {
        terminate!();
    }

    let query_head = ABSOLUTE_POS / head_dim;
    let output_dim = ABSOLUTE_POS % head_dim;
    let kv_head = query_head / groups;
    let mut running_max = -3.402_823_5e38_f32;
    let mut running_sum = 0.0_f32;
    let mut running_output = 0.0_f32;

    #[unroll]
    for page in 0..keys.len() {
        let key = &keys[page];
        let value = &values[page];
        for token in 0..key.shape(2) {
            let mut score = 0.0_f32;
            for dim in 0..head_dim {
                let query_offset = query_head * query.stride(1) + dim * query.stride(3);
                let key_offset =
                    kv_head * key.stride(1) + token * key.stride(2) + dim * key.stride(3);
                score += f32::cast_from(query[query_offset]) * f32::cast_from(key[key_offset]);
            }
            score *= scale;
            let new_max = if score > running_max {
                score
            } else {
                running_max
            };
            let previous_scale = (running_max - new_max).exp();
            let weight = (score - new_max).exp();
            let value_offset =
                kv_head * value.stride(1) + token * value.stride(2) + output_dim * value.stride(3);
            running_sum = running_sum * previous_scale + weight;
            running_output =
                running_output * previous_scale + weight * f32::cast_from(value[value_offset]);
            running_max = new_max;
        }
    }

    let output_offset = query_head * output.stride(1) + output_dim * output.stride(3);
    output[output_offset] = F::cast_from(running_output / running_sum);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_accelerator_target() {
        let error = CubePagedAttention::new(KernelTarget::Portable).unwrap_err();
        assert!(error.message().contains("WGPU or CUDA"));
    }

    #[test]
    fn descriptors_distinguish_runtime_targets() {
        let wgpu = CubePagedAttention::new(KernelTarget::Wgpu)
            .unwrap()
            .descriptor();
        let cuda = CubePagedAttention::new(KernelTarget::Cuda)
            .unwrap()
            .descriptor();

        assert_eq!(wgpu.name(), "cubecl-wgpu-paged-attention");
        assert_eq!(wgpu.target(), KernelTarget::Wgpu);
        assert_eq!(cuda.name(), "cubecl-cuda-paged-attention");
        assert_eq!(cuda.target(), KernelTarget::Cuda);
    }
}
