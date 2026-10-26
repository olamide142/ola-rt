use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use burn::tensor::Tensor;

use crate::{KernelDescriptor, KernelError, PagedAttentionInput, PagedAttentionKernel};

#[derive(Debug)]
struct KernelExecutionMetricsInner {
    measure_time: bool,
    calls: AtomicU64,
    host_duration_nanos: AtomicU64,
}

/// Shared counters for synchronous calls to one selected kernel.
///
/// The duration covers time spent inside [`PagedAttentionKernel::execute`] on
/// the host. It includes device work only when the implementation synchronizes
/// before returning; later tensor read-back remains part of the wider model
/// invocation timing.
#[derive(Clone, Debug)]
pub struct KernelExecutionMetrics {
    inner: Arc<KernelExecutionMetricsInner>,
}

impl Default for KernelExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelExecutionMetrics {
    /// Creates an empty, independently shared measurement.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(KernelExecutionMetricsInner {
                measure_time: true,
                calls: AtomicU64::new(0),
                host_duration_nanos: AtomicU64::new(0),
            }),
        }
    }

    /// Creates deterministic call counters without reading a wall clock.
    pub fn without_timing() -> Self {
        Self {
            inner: Arc::new(KernelExecutionMetricsInner {
                measure_time: false,
                calls: AtomicU64::new(0),
                host_duration_nanos: AtomicU64::new(0),
            }),
        }
    }

    /// Returns the current aggregate without resetting it.
    pub fn snapshot(&self) -> KernelExecutionSnapshot {
        KernelExecutionSnapshot {
            calls: self.inner.calls.load(Ordering::Relaxed),
            host_duration_nanos: self
                .inner
                .measure_time
                .then(|| self.inner.host_duration_nanos.load(Ordering::Relaxed)),
        }
    }

    fn measures_time(&self) -> bool {
        self.inner.measure_time
    }

    fn record(&self, host_duration_nanos: Option<u64>) {
        let _ = self
            .inner
            .calls
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |calls| {
                Some(calls.saturating_add(1))
            });
        if let Some(host_duration_nanos) = host_duration_nanos {
            let _ = self.inner.host_duration_nanos.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |duration| Some(duration.saturating_add(host_duration_nanos)),
            );
        }
    }
}

/// Aggregate observed while executing one model invocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelExecutionSnapshot {
    /// Number of synchronous kernel calls.
    pub calls: u64,
    /// Host wall time spent inside those calls, in nanoseconds, when enabled.
    pub host_duration_nanos: Option<u64>,
}

pub(crate) struct MeasuredPagedAttentionKernel {
    inner: Box<dyn PagedAttentionKernel>,
    metrics: KernelExecutionMetrics,
}

impl MeasuredPagedAttentionKernel {
    pub(crate) fn new(
        inner: Box<dyn PagedAttentionKernel>,
        metrics: KernelExecutionMetrics,
    ) -> Self {
        Self { inner, metrics }
    }
}

impl PagedAttentionKernel for MeasuredPagedAttentionKernel {
    fn descriptor(&self) -> KernelDescriptor {
        self.inner.descriptor()
    }

    fn execute(&self, input: PagedAttentionInput) -> Result<Tensor<4>, KernelError> {
        let started = self.metrics.measures_time().then(Instant::now);
        let result = self.inner.execute(input);
        let nanos =
            started.map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        self.metrics.record(nanos);
        result
    }

    fn execute_batch(
        &self,
        inputs: Vec<PagedAttentionInput>,
    ) -> Result<Vec<Tensor<4>>, KernelError> {
        let started = self.metrics.measures_time().then(Instant::now);
        let result = self.inner.execute_batch(inputs);
        let nanos =
            started.map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        self.metrics.record(nanos);
        result
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::*;

    #[test]
    fn clockless_metrics_count_calls_without_recording_wall_time() {
        let metrics = KernelExecutionMetrics::without_timing();

        metrics.record(None);

        assert_eq!(
            metrics.snapshot(),
            KernelExecutionSnapshot {
                calls: 1,
                host_duration_nanos: None,
            }
        );
    }

    #[test]
    fn timed_metrics_accumulate_host_duration() {
        let metrics = KernelExecutionMetrics::new();

        metrics.record(Some(7));
        metrics.record(Some(11));

        assert_eq!(
            metrics.snapshot(),
            KernelExecutionSnapshot {
                calls: 2,
                host_duration_nanos: Some(18),
            }
        );
    }

    #[test]
    fn batched_execution_records_one_dispatch() {
        let device = Device::default();
        let query = Tensor::from_data(TensorData::new(vec![1.0_f32], [1, 1, 1, 1]), &device);
        let key = Tensor::from_data(TensorData::new(vec![1.0_f32], [1, 1, 1, 1]), &device);
        let value = Tensor::from_data(TensorData::new(vec![2.0_f32], [1, 1, 1, 1]), &device);
        let config = crate::PagedAttentionConfig::new(1, 1, 1).unwrap();
        let input = crate::PagedAttentionInput::new(
            query,
            vec![crate::PagedKvPage::new(key, value)],
            config,
        )
        .unwrap();
        let metrics = KernelExecutionMetrics::without_timing();
        let kernel = MeasuredPagedAttentionKernel::new(
            Box::new(crate::PortablePagedAttention::new()),
            metrics.clone(),
        );

        kernel
            .execute_batch(vec![input.clone(), input])
            .expect("portable batch fallback must execute");

        assert_eq!(metrics.snapshot().calls, 1);
    }
}
