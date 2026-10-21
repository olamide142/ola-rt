use burn::{
    backend::{Dispatch, DispatchDevice, backend_extension, tensor::FloatTensor},
    tensor::{DType, Tensor as BurnTensor},
};
use burn_backend::{Backend, Shape, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    CubeBackend, CubeDevice, cubecl, kernel::untile, ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    KernelCapability, KernelDType, KernelDescriptor, KernelError, KernelOperation, KernelTarget,
    PagedAttentionInput, PagedAttentionKernel, PagedAttentionProvider,
};

/// Backend extension routed through Burn's CubeCL and Fusion backends.
#[backend_extension(Cube, Fusion)]
pub trait PagedAttentionOps: Backend {
    /// Computes one-token attention from contiguous query, key and value tensors.
    #[fusion(dtype = query, shape = query)]
    fn paged_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        groups: usize,
    ) -> FloatTensor<Self>;
}

impl PagedAttentionOps for CubeBackend {
    fn paged_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        groups: usize,
    ) -> FloatTensor<Self> {
        launch_paged_attention(untile(query), untile(key), untile(value), groups)
    }
}

/// Single-sequence decode attention implemented with CubeCL.
#[derive(Clone, Copy, Debug)]
pub struct CubePagedAttention {
    target: KernelTarget,
}

impl CubePagedAttention {
    /// Creates a CubeCL kernel for a WGPU or CUDA execution target.
    pub fn new(target: KernelTarget) -> Result<Self, KernelError> {
        match target {
            KernelTarget::Wgpu | KernelTarget::Cuda => Ok(Self { target }),
            #[cfg(feature = "cubecl-cpu")]
            KernelTarget::Cpu => Ok(Self { target }),
            _ => Err(KernelError::new(
                "CubeCL paged attention currently supports WGPU or CUDA targets",
            )),
        }
    }

    /// Reports whether this implementation can consume the validated input.
    ///
    /// The first implementation accepts f32 inference tensors on the selected
    /// GPU runtime. Other devices and dtypes should retain the portable path.
    pub fn supports(&self, input: &PagedAttentionInput) -> bool {
        if input.dtype() != DType::F32 {
            return false;
        }
        match (self.target, input.query().device().as_dispatch()) {
            (KernelTarget::Wgpu, DispatchDevice::Cube(CubeDevice::Wgpu(_)))
            | (KernelTarget::Cuda, DispatchDevice::Cube(CubeDevice::Cuda(_))) => true,
            #[cfg(feature = "cubecl-cpu")]
            (KernelTarget::Cpu, DispatchDevice::Cube(CubeDevice::Cpu(_))) => true,
            _ => false,
        }
    }
}

impl PagedAttentionKernel for CubePagedAttention {
    fn descriptor(&self) -> KernelDescriptor {
        let name = match self.target {
            KernelTarget::Wgpu => "cubecl-wgpu-paged-attention",
            KernelTarget::Cuda => "cubecl-cuda-paged-attention",
            #[cfg(feature = "cubecl-cpu")]
            KernelTarget::Cpu => "cubecl-cpu-paged-attention",
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
        let mut keys = Vec::with_capacity(pages.len());
        let mut values = Vec::with_capacity(pages.len());
        for page in pages {
            keys.push(page.key().clone());
            values.push(page.value().clone());
        }

        // CubeCL 0.11 sequences cannot contain tensor arguments. Keep the page
        // API at the Ola boundary and pack the device tensors before dispatch.
        let key = BurnTensor::cat(keys, 2);
        let value = BurnTensor::cat(values, 2);
        let output = <Dispatch as PagedAttentionOps>::paged_attention(
            query.into_dispatch(),
            key.into_dispatch(),
            value.into_dispatch(),
            config.groups(),
        );
        Ok(BurnTensor::<4>::from_dispatch(output))
    }
}

impl PagedAttentionProvider for CubePagedAttention {
    fn capability(&self) -> KernelCapability {
        KernelCapability::new(self.descriptor(), [self.target], [KernelDType::F32])
    }

    fn priority(&self) -> u16 {
        100
    }

    fn create(&self) -> Box<dyn PagedAttentionKernel> {
        Box::new(*self)
    }
}

fn launch_paged_attention(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    groups: usize,
) -> CubeTensor {
    let client = query.client.clone();
    let output = empty_device_dtype(
        client.clone(),
        query.device.clone(),
        query.meta.shape().clone(),
        query.dtype,
    );
    let [_, heads, _, dim]: [usize; 4] = query
        .meta
        .shape()
        .dims()
        .try_into()
        .expect("validated query has rank four");
    let [_, _, tokens, _]: [usize; 4] = key
        .meta
        .shape()
        .dims()
        .try_into()
        .expect("validated key pages have rank four");
    let score_shape = Shape::new([1, heads, 1, tokens]);
    let scores = empty_device_dtype(
        client.clone(),
        query.device.clone(),
        score_shape,
        query.dtype,
    );
    let score_items = heads * tokens;
    let score_cube_dim = CubeDim::new(&client, score_items);
    let score_cube_count = calculate_cube_count_elemwise(&client, score_items, score_cube_dim);
    paged_attention_scores::launch(
        &client,
        score_cube_count,
        score_cube_dim,
        query.into_tensor_arg(),
        key.into_tensor_arg(),
        scores.clone().into_tensor_arg(),
        groups,
        dtype_to_storage_type(scores.dtype),
    );

    let work_items = heads * dim;
    let cube_dim = CubeDim::new(&client, work_items);
    let cube_count = calculate_cube_count_elemwise(&client, work_items, cube_dim);

    paged_attention::launch(
        &client,
        cube_count,
        cube_dim,
        scores.into_tensor_arg(),
        value.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        groups,
        dtype_to_storage_type(output.dtype),
    );
    output
}

/// Computes each query-head/key-token score once before value accumulation.
#[cube(launch)]
fn paged_attention_scores<F: Float>(
    query: &Tensor<F>,
    key: &Tensor<F>,
    scores: &mut Tensor<F>,
    groups: usize,
    #[define(F)] _dtype: ElemType,
) {
    let head_dim = query.shape(3);
    let tokens = key.shape(2);
    let score_elements = query.shape(1) * tokens;
    if ABSOLUTE_POS >= score_elements {
        terminate!();
    }

    let query_head = ABSOLUTE_POS / tokens;
    let token = ABSOLUTE_POS % tokens;
    let kv_head = query_head / groups;
    let scale = f32::cast_from(head_dim).sqrt().recip();
    let mut score = f32::new(0.0);
    for dim in 0..head_dim {
        let query_offset = query_head * query.stride(1) + dim * query.stride(3);
        let key_offset = kv_head * key.stride(1) + token * key.stride(2) + dim * key.stride(3);
        score = score + f32::cast_from(query[query_offset]) * f32::cast_from(key[key_offset]);
    }
    let score_offset = query_head * scores.stride(1) + token * scores.stride(3);
    scores[score_offset] = F::cast_from(score * scale);
}

/// Applies online softmax to precomputed scores and accumulates one value dim.
#[cube(launch)]
fn paged_attention<F: Float>(
    scores: &Tensor<F>,
    value: &Tensor<F>,
    output: &mut Tensor<F>,
    groups: usize,
    #[define(F)] _dtype: ElemType,
) {
    let head_dim = value.shape(3);
    let output_elements = scores.shape(1) * head_dim;
    if ABSOLUTE_POS >= output_elements {
        terminate!();
    }

    let query_head = ABSOLUTE_POS / head_dim;
    let output_dim = ABSOLUTE_POS % head_dim;
    let kv_head = query_head / groups;
    let mut running_max = f32::new(-3.402_823_5e38);
    let mut running_sum = f32::new(0.0);
    let mut running_output = f32::new(0.0);

    for token in 0..scores.shape(3) {
        let score_offset = query_head * scores.stride(1) + token * scores.stride(3);
        let score = f32::cast_from(scores[score_offset]);
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

    let output_offset = query_head * output.stride(1) + output_dim * output.stride(3);
    output[output_offset] = F::cast_from(running_output / running_sum);
}

#[cfg(all(test, any(feature = "cubecl-cpu", feature = "wgpu")))]
mod tests {
    #[cfg(feature = "wgpu")]
    use std::time::{Duration, Instant};

    #[cfg(feature = "wgpu")]
    use burn::tensor::DeviceKind;
    use burn::tensor::{Device, Tensor, TensorData};

    use super::*;
    use crate::{PagedAttentionConfig, PagedKvPage, PortablePagedAttention};

    #[test]
    #[cfg(feature = "cubecl-cpu")]
    fn cubecl_cpu_matches_portable_across_pages_and_grouped_heads() {
        assert_matches(Device::cpu(), KernelTarget::Cpu);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn cubecl_wgpu_matches_portable_across_pages_and_grouped_heads() {
        let device = Device::wgpu(DeviceKind::IntegratedGpu(0));
        assert_matches(device.clone(), KernelTarget::Wgpu);
        assert_input_matches(benchmark_input(&device, 2), KernelTarget::Wgpu, 1e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "manual WGPU microbenchmark; reports output-readback latency"]
    fn benchmark_cubecl_wgpu_against_portable_attention() {
        const WARMUP: usize = 2;
        const RUNS: usize = 8;
        let device = Device::wgpu(DeviceKind::IntegratedGpu(0));
        let one_page = benchmark_input(&device, 1);
        let two_pages = benchmark_input(&device, 2);
        let cubecl = CubePagedAttention::new(KernelTarget::Wgpu).unwrap();
        let portable = PortablePagedAttention::new();

        let cubecl_one_page = measure(&cubecl, &one_page, WARMUP, RUNS);
        let cubecl_two_pages = measure(&cubecl, &two_pages, WARMUP, RUNS);
        let portable_one_page = measure(&portable, &one_page, WARMUP, RUNS);
        let portable_two_pages = measure(&portable, &two_pages, WARMUP, RUNS);
        eprintln!(
            "WGPU attention (32 query heads, 8 KV heads, dim 64, 256 tokens): CubeCL one/two pages {:?}/{:?}, portable {:?}/{:?}; includes synchronization and output readback",
            cubecl_one_page, cubecl_two_pages, portable_one_page, portable_two_pages,
        );
    }

    fn assert_matches(device: Device, target: KernelTarget) {
        let query = Tensor::from_data(
            TensorData::new(
                vec![
                    0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0.8, -0.9, 1.0, 0.2, -0.3, 0.5, 0.4,
                    -0.1, 0.9,
                ],
                [1, 4, 1, 4],
            ),
            &device,
        );
        let key_a = Tensor::from_data(
            TensorData::new(
                vec![
                    0.2_f32, 0.1, -0.3, 0.4, 0.5, -0.2, 0.7, 0.3, -0.6, 0.8, 0.1, -0.4, 0.9, 0.2,
                    -0.1, 0.6, -0.7, 0.3, 0.4, 0.5, 0.1, -0.8, 0.2, 0.7,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value_a = Tensor::from_data(
            TensorData::new(
                vec![
                    0.3_f32, -0.1, 0.8, 0.2, -0.4, 0.7, 0.1, -0.5, 0.6, 0.2, -0.3, 0.9, -0.7, 0.4,
                    0.5, 0.1, 0.2, 0.8, -0.6, 0.3, 0.9, -0.2, 0.4, -0.1,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let key_b = Tensor::from_data(
            TensorData::new(
                vec![0.6_f32, -0.5, 0.2, 0.1, -0.3, 0.9, 0.4, -0.7],
                [1, 2, 1, 4],
            ),
            &device,
        );
        let value_b = Tensor::from_data(
            TensorData::new(
                vec![-0.2_f32, 0.5, 0.7, -0.4, 0.8, 0.1, -0.9, 0.3],
                [1, 2, 1, 4],
            ),
            &device,
        );
        let config = PagedAttentionConfig::new(4, 2, 4).unwrap();
        let input = PagedAttentionInput::new(
            query,
            vec![
                PagedKvPage::new(key_a, value_a),
                PagedKvPage::new(key_b, value_b),
            ],
            config,
        )
        .unwrap();
        assert_input_matches(input, target, 1e-5);
    }

    fn assert_input_matches(input: PagedAttentionInput, target: KernelTarget, tolerance: f32) {
        let expected = PortablePagedAttention::new()
            .execute(input.clone())
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let actual = CubePagedAttention::new(target)
            .unwrap()
            .execute(input)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "output {index} differs by {error} (tolerance {tolerance})"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn benchmark_input(device: &Device, page_count: usize) -> PagedAttentionInput {
        const QUERY_HEADS: usize = 32;
        const KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 64;
        const TOTAL_TOKENS: usize = 256;
        let tokens_per_page = TOTAL_TOKENS / page_count;
        let make_values = |len: usize, phase: usize| {
            (0..len)
                .map(|index| (((index + phase) % 101) as f32 - 50.0) * 0.01)
                .collect::<Vec<_>>()
        };
        let query = Tensor::from_data(
            TensorData::new(
                make_values(QUERY_HEADS * HEAD_DIM, 3),
                [1, QUERY_HEADS, 1, HEAD_DIM],
            ),
            device,
        );
        let elements_per_page = KV_HEADS * tokens_per_page * HEAD_DIM;
        let key_values = make_values(KV_HEADS * TOTAL_TOKENS * HEAD_DIM, 0);
        let value_values = make_values(KV_HEADS * TOTAL_TOKENS * HEAD_DIM, 5);
        let pages = (0..page_count)
            .map(|page| {
                let start = page * elements_per_page;
                let end = start + elements_per_page;
                let shape = [1, KV_HEADS, tokens_per_page, HEAD_DIM];
                let key = Tensor::from_data(
                    TensorData::new(key_values[start..end].to_vec(), shape),
                    device,
                );
                let value = Tensor::from_data(
                    TensorData::new(value_values[start..end].to_vec(), shape),
                    device,
                );
                PagedKvPage::new(key, value)
            })
            .collect();
        PagedAttentionInput::new(
            query,
            pages,
            PagedAttentionConfig::new(QUERY_HEADS, KV_HEADS, HEAD_DIM).unwrap(),
        )
        .unwrap()
    }

    #[cfg(feature = "wgpu")]
    fn measure(
        kernel: &dyn PagedAttentionKernel,
        input: &PagedAttentionInput,
        warmup: usize,
        runs: usize,
    ) -> Duration {
        for _ in 0..warmup {
            let _ = kernel
                .execute(input.clone())
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = kernel
                .execute(input.clone())
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
        }
        start.elapsed() / runs as u32
    }
}
