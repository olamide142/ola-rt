use burn::{
    backend::{
        Dispatch, DispatchDevice, backend_extension,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Int as BurnInt, Tensor as BurnTensor, TensorData},
};
use burn_backend::{Backend, cubecl::dtype_to_storage_type};
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
        block_table: IntTensor<Self>,
        token_counts: IntTensor<Self>,
        groups: usize,
    ) -> FloatTensor<Self>;
}

impl PagedAttentionOps for CubeBackend {
    fn paged_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        block_table: IntTensor<Self>,
        token_counts: IntTensor<Self>,
        groups: usize,
    ) -> FloatTensor<Self> {
        launch_paged_attention(
            untile(query),
            untile(key),
            untile(value),
            untile(block_table),
            untile(token_counts),
            groups,
        )
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
        if input.blocks().is_none() {
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

    /// Returns whether two supported inputs can share one physical launch.
    fn can_share_batch(first: &PagedAttentionInput, input: &PagedAttentionInput) -> bool {
        let Some(first_blocks) = first.blocks() else {
            return false;
        };
        let Some(blocks) = input.blocks() else {
            return false;
        };
        input.config() == first.config()
            && input.dtype() == first.dtype()
            && input.query().device() == first.query().device()
            && blocks.key().dims() == first_blocks.key().dims()
            && blocks.value().dims() == first_blocks.value().dims()
            && blocks.tokens_per_block() == first_blocks.tokens_per_block()
            && blocks.pool_id() == first_blocks.pool_id()
    }

    /// Executes one homogeneous group as a single padded-table dispatch.
    fn execute_shareable_batch(
        &self,
        inputs: Vec<PagedAttentionInput>,
    ) -> Result<Vec<BurnTensor<4>>, KernelError> {
        let mut queries = Vec::with_capacity(inputs.len());
        let mut blocks = Vec::with_capacity(inputs.len());
        let mut config = None;
        for input in inputs {
            if !self.supports(&input) {
                return Err(KernelError::new("CubeCL does not support a batched input"));
            }
            let (query, _pages, physical, input_config) = input.into_parts();
            let physical = physical.expect("supports requires physical KV blocks");
            if let Some(expected) = config {
                if expected != input_config {
                    return Err(KernelError::new(
                        "batched paged attention inputs must share one configuration",
                    ));
                }
            } else {
                config = Some(input_config);
            }
            queries.push(query);
            blocks.push(physical);
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let config = config.expect("non-empty batch has a configuration");
        let key = blocks[0].key().clone();
        let value = blocks[0].value().clone();
        let tokens_per_block = blocks[0].tokens_per_block();
        let max_blocks = blocks
            .iter()
            .map(|block| block.block_table().len())
            .max()
            .expect("non-empty batch has blocks");
        let mut table_values = Vec::with_capacity(blocks.len() * max_blocks);
        let mut token_values = Vec::with_capacity(blocks.len());
        for block in &blocks {
            if block.key().dims() != key.dims()
                || block.value().dims() != value.dims()
                || block.tokens_per_block() != tokens_per_block
                || block.key().device() != key.device()
            {
                return Err(KernelError::new(
                    "batched paged attention inputs must share one physical KV pool",
                ));
            }
            let ids = block.block_table().block_ids();
            table_values.extend(ids.iter().map(|id| *id as i64));
            table_values.extend(std::iter::repeat_n(0_i64, max_blocks - ids.len()));
            token_values.push(block.token_count() as i64);
        }
        let device = queries[0].device();
        let query = BurnTensor::cat(queries, 0);
        let block_table: BurnTensor<2, BurnInt> = BurnTensor::from_data(
            TensorData::new(table_values, [blocks.len(), max_blocks]),
            &device,
        );
        let token_counts: BurnTensor<1, BurnInt> =
            BurnTensor::from_data(TensorData::new(token_values, [blocks.len()]), &device);
        let output = <Dispatch as PagedAttentionOps>::paged_attention(
            query.into_dispatch(),
            key.into_dispatch(),
            value.into_dispatch(),
            block_table.into_dispatch(),
            token_counts.into_dispatch(),
            config.groups(),
        );
        let output = BurnTensor::<4>::from_dispatch(output);
        Ok((0..blocks.len())
            .map(|index| {
                output.clone().slice([
                    index..index + 1,
                    0..config.query_heads(),
                    0..1,
                    0..config.head_dim(),
                ])
            })
            .collect())
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

        let (query, _pages, blocks, config) = input.into_parts();
        let blocks = blocks.expect("supports requires physical KV blocks");
        let (key, value, block_table, _tokens_per_block, token_count) = blocks.into_parts();
        let block_count = block_table.dims()[0];
        let block_table = block_table.reshape([1, block_count]);
        let token_counts: BurnTensor<1, BurnInt> = BurnTensor::from_data(
            TensorData::new(vec![token_count as i64], [1]),
            &query.device(),
        );
        let output = <Dispatch as PagedAttentionOps>::paged_attention(
            query.into_dispatch(),
            key.into_dispatch(),
            value.into_dispatch(),
            block_table.into_dispatch(),
            token_counts.into_dispatch(),
            config.groups(),
        );
        Ok(BurnTensor::<4>::from_dispatch(output))
    }

    fn execute_batch(
        &self,
        inputs: Vec<PagedAttentionInput>,
    ) -> Result<Vec<BurnTensor<4>>, KernelError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut groups: Vec<Vec<(usize, PagedAttentionInput)>> = Vec::new();
        let mut outputs: Vec<Option<BurnTensor<4>>> = (0..inputs.len()).map(|_| None).collect();
        for (index, input) in inputs.into_iter().enumerate() {
            if !self.supports(&input) {
                outputs[index] = Some(self.execute(input)?);
                continue;
            }
            let group = groups
                .iter_mut()
                .find(|group| Self::can_share_batch(&group[0].1, &input));
            if let Some(group) = group {
                group.push((index, input));
            } else {
                groups.push(vec![(index, input)]);
            }
        }
        for group in groups {
            let indices = group.iter().map(|(index, _)| *index).collect::<Vec<_>>();
            let group_inputs = group
                .into_iter()
                .map(|(_, input)| input)
                .collect::<Vec<_>>();
            let group_outputs = self.execute_shareable_batch(group_inputs)?;
            for (index, output) in indices.into_iter().zip(group_outputs) {
                outputs[index] = Some(output);
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| KernelError::new("batched attention produced no output"))
            })
            .collect()
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
    block_table: CubeTensor,
    token_counts: CubeTensor,
    groups: usize,
) -> CubeTensor {
    let client = query.client.clone();
    let output = empty_device_dtype(
        client.clone(),
        query.device.clone(),
        query.meta.shape().clone(),
        query.dtype,
    );
    let [batch, heads, _, dim] = query.meta.shape().dims();
    let work_items = batch * heads * dim;
    let cube_dim = CubeDim::new(&client, work_items);
    let cube_count = calculate_cube_count_elemwise(&client, work_items, cube_dim);

    paged_attention::launch(
        &client,
        cube_count,
        cube_dim,
        query.into_tensor_arg(),
        key.into_tensor_arg(),
        value.into_tensor_arg(),
        block_table.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        token_counts.into_tensor_arg(),
        groups,
        dtype_to_storage_type(output.dtype),
        dtype_to_storage_type(block_table.dtype),
    );
    output
}

/// Traverses physical pages and applies online softmax in one dispatch.
#[cube(launch)]
fn paged_attention<F: Float, I: Int>(
    query: &Tensor<F>,
    key: &Tensor<F>,
    value: &Tensor<F>,
    block_table: &Tensor<I>,
    output: &mut Tensor<F>,
    token_counts: &Tensor<I>,
    groups: usize,
    #[define(F)] _dtype: ElemType,
    #[define(I)] _int_dtype: ElemType,
) {
    let head_dim = value.shape(3);
    let output_elements = query.shape(0) * query.shape(1) * head_dim;
    if ABSOLUTE_POS >= output_elements {
        terminate!();
    }

    let request = ABSOLUTE_POS / (query.shape(1) * head_dim);
    let head_position = ABSOLUTE_POS % (query.shape(1) * head_dim);
    let query_head = head_position / head_dim;
    let output_dim = head_position % head_dim;
    let kv_head = query_head / groups;
    let tokens_per_block = key.shape(2);
    let token_count = usize::cast_from(token_counts[request]);
    let scale = f32::cast_from(head_dim).sqrt().recip();
    let mut running_max = f32::new(-3.402_823_5e38);
    let mut running_sum = f32::new(0.0);
    let mut running_output = f32::new(0.0);

    for token in 0..token_count {
        let logical_block = token / tokens_per_block;
        let block_offset = token % tokens_per_block;
        let table_offset = request * block_table.stride(0) + logical_block * block_table.stride(1);
        let physical_block = usize::cast_from(block_table[table_offset]);
        let mut score = f32::new(0.0);
        for dim in 0..head_dim {
            let query_offset =
                request * query.stride(0) + query_head * query.stride(1) + dim * query.stride(3);
            let key_offset = physical_block * key.stride(0)
                + kv_head * key.stride(1)
                + block_offset * key.stride(2)
                + dim * key.stride(3);
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
        let value_offset = physical_block * value.stride(0)
            + kv_head * value.stride(1)
            + block_offset * value.stride(2)
            + output_dim * value.stride(3);
        running_sum = running_sum * previous_scale + weight;
        running_output =
            running_output * previous_scale + weight * f32::cast_from(value[value_offset]);
        running_max = new_max;
    }

    let output_offset =
        request * output.stride(0) + query_head * output.stride(1) + output_dim * output.stride(3);
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
    use crate::{
        PagedAttentionConfig, PagedKvBlockTable, PagedKvBlocks, PagedKvPage, PortablePagedAttention,
    };

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
        let cubecl_two_request_batch = measure_batch(
            &cubecl,
            &[two_pages.clone(), two_pages.clone()],
            WARMUP,
            RUNS,
        );
        eprintln!(
            "WGPU attention (32 query heads, 8 KV heads, dim 64, 256 tokens): CubeCL one/two pages {:?}/{:?}, two-request batch {:?}, portable {:?}/{:?}; includes synchronization and output readback",
            cubecl_one_page,
            cubecl_two_pages,
            cubecl_two_request_batch,
            portable_one_page,
            portable_two_pages,
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
        assert_input_matches(attach_physical_blocks(input), target, 1e-5);
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
            .execute(input.clone())
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

        let batched = CubePagedAttention::new(target)
            .unwrap()
            .execute_batch(vec![input.clone(), input.clone()])
            .unwrap();
        assert_eq!(batched.len(), 2);
        let first = batched[0].clone().into_data().try_to_vec::<f32>().unwrap();
        let second = batched[1].clone().into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(first, second);

        // The second request has fewer logical blocks but shares the same
        // physical pool geometry. This is the ragged request case: each row
        // must stop at its own token count rather than the padded table width.
        let short = PagedAttentionInput::new(
            input.query().clone(),
            vec![input.pages()[0].clone()],
            input.config(),
        )
        .unwrap();
        let short = attach_physical_blocks(short);
        let expected_short = PortablePagedAttention::new()
            .execute(short.clone())
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let ragged = CubePagedAttention::new(target)
            .unwrap()
            .execute_batch(vec![input.clone(), short])
            .unwrap();
        let actual_short = ragged[1].clone().into_data().try_to_vec::<f32>().unwrap();
        for (actual, expected) in actual_short.into_iter().zip(expected_short) {
            assert!((actual - expected).abs() <= tolerance);
        }

        let incompatible_page = PagedKvPage::new(
            input.pages()[0]
                .key()
                .clone()
                .slice([0..1, 0..2, 0..2, 0..4]),
            input.pages()[0]
                .value()
                .clone()
                .slice([0..1, 0..2, 0..2, 0..4]),
        );
        let incompatible = attach_physical_blocks(
            PagedAttentionInput::new(
                input.query().clone(),
                vec![incompatible_page],
                input.config(),
            )
            .unwrap(),
        );
        let expected_incompatible = PortablePagedAttention::new()
            .execute(incompatible.clone())
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let independent_source = PagedAttentionInput::new(
            input.query().clone(),
            input.pages().to_vec(),
            input.config(),
        )
        .unwrap();
        let independent_pool = attach_physical_blocks_with_pool_id(independent_source, 8);
        let grouped = CubePagedAttention::new(target)
            .unwrap()
            .execute_batch(vec![input.clone(), incompatible])
            .unwrap();
        let actual_incompatible = grouped[1].clone().into_data().try_to_vec::<f32>().unwrap();
        for (actual, expected) in actual_incompatible.into_iter().zip(expected_incompatible) {
            assert!((actual - expected).abs() <= tolerance);
        }

        // Matching shapes are not sufficient to fuse requests: independently
        // allocated physical pools must remain isolated even when their
        // layouts happen to be identical.
        let expected_independent = PortablePagedAttention::new()
            .execute(independent_pool.clone())
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let isolated = CubePagedAttention::new(target)
            .unwrap()
            .execute_batch(vec![input, independent_pool])
            .unwrap();
        let actual_independent = isolated[1].clone().into_data().try_to_vec::<f32>().unwrap();
        for (actual, expected) in actual_independent.into_iter().zip(expected_independent) {
            assert!((actual - expected).abs() <= tolerance);
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
        attach_physical_blocks(
            PagedAttentionInput::new(
                query,
                pages,
                PagedAttentionConfig::new(QUERY_HEADS, KV_HEADS, HEAD_DIM).unwrap(),
            )
            .unwrap(),
        )
    }

    fn attach_physical_blocks(input: PagedAttentionInput) -> PagedAttentionInput {
        attach_physical_blocks_with_pool_id(input, 7)
    }

    fn attach_physical_blocks_with_pool_id(
        input: PagedAttentionInput,
        pool_id: u64,
    ) -> PagedAttentionInput {
        let device = input.query().device();
        let config = input.config();
        let tokens_per_block = input
            .pages()
            .iter()
            .map(|page| page.key().dims()[2])
            .max()
            .unwrap();
        let physical_blocks = (input.pages().len() + 1).max(3);
        let shape = [
            physical_blocks,
            config.key_value_heads(),
            tokens_per_block,
            config.head_dim(),
        ];
        let mut key = Tensor::zeros(shape, &device);
        let mut value = Tensor::zeros(shape, &device);
        let mut block_ids = Vec::with_capacity(input.pages().len());
        let mut token_count = 0;
        for (logical, page) in input.pages().iter().enumerate() {
            let physical = physical_blocks - 1 - logical;
            let used = page.key().dims()[2];
            key = key.slice_assign(
                [
                    physical..physical + 1,
                    0..config.key_value_heads(),
                    0..used,
                    0..config.head_dim(),
                ],
                page.key().clone(),
            );
            value = value.slice_assign(
                [
                    physical..physical + 1,
                    0..config.key_value_heads(),
                    0..used,
                    0..config.head_dim(),
                ],
                page.value().clone(),
            );
            block_ids.push(physical);
            token_count += used;
        }
        let block_table = PagedKvBlockTable::new(block_ids, physical_blocks, &device).unwrap();
        let blocks = PagedKvBlocks::new_with_pool_id(
            key,
            value,
            block_table,
            tokens_per_block,
            token_count,
            pool_id,
        )
        .unwrap();
        input.with_blocks(blocks).unwrap()
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

    #[cfg(feature = "wgpu")]
    fn measure_batch(
        kernel: &dyn PagedAttentionKernel,
        inputs: &[PagedAttentionInput],
        warmup: usize,
        runs: usize,
    ) -> Duration {
        for _ in 0..warmup {
            let _ = kernel
                .execute_batch(inputs.to_vec())
                .unwrap()
                .into_iter()
                .flat_map(|output| output.into_data().try_to_vec::<f32>().unwrap())
                .collect::<Vec<_>>();
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = kernel
                .execute_batch(inputs.to_vec())
                .unwrap()
                .into_iter()
                .flat_map(|output| output.into_data().try_to_vec::<f32>().unwrap())
                .collect::<Vec<_>>();
        }
        start.elapsed() / runs as u32
    }
}
