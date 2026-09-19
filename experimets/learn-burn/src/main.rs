#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::{
            vision::{MnistDataset, MnistItem},
            Dataset,
        },
    },
    nn::{
        loss::CrossEntropyLossConfig,
        Linear,
        LinearConfig,
        Relu,
    },
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{
        backend::AutodiffBackend,
        Tensor,
    },
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput,
        InferenceStep,
        Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
    },
};

// ============================================================
// MODEL
// ============================================================

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 512)]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // MNIST image:
            //
            // 28 x 28 = 784 pixels
            //
            // So each image becomes:
            //
            // [784]
            //
            hidden: LinearConfig::new(784, self.hidden_size).init(device),

            // Output 10 logits:
            //
            // digit 0
            // digit 1
            // ...
            // digit 9
            //
            output: LinearConfig::new(self.hidden_size, 10).init(device),

            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        /*
            Input shape:

                [batch_size, 28, 28]

            Example:

                [64, 28, 28]

            We flatten each image:

                [64, 28, 28]

                        ↓

                [64, 784]
        */

        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, height * width]);

        /*
            First layer:

                [64, 784]

                    ×

                weights

                    ↓

                [64, 512]
        */

        let x = self.hidden.forward(x);

        /*
            ReLU activation:

                max(0, x)
        */

        let x = self.activation.forward(x);

        /*
            Final layer:

                [64, 512]

                    ↓

                [64, 10]

            Each row contains 10 logits.
        */

        self.output.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(images);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }
}

// ============================================================
// BATCH
// ============================================================

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

// ============================================================
// BATCHER
// ============================================================

#[derive(Clone, Debug, Default)]
pub struct MnistBatcher {}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        /*
            Every MnistItem contains:

                image: [[u8; 28]; 28]
                label: u8

            Suppose:

                batch_size = 64

            We receive:

                Vec<MnistItem>

            containing 64 MNIST images.
        */

        let images = items
            .iter()
            .map(|item| {
                /*
                    Convert image pixels into a tensor.

                    Original pixel range:

                        0..255

                    We normalize them into:

                        0.0..1.0
                */

                Tensor::<B, 2>::from_data(
                    TensorData::from(item.image).convert::<B::FloatElem>(),
                    device,
                )
                / 255.0
            })
            .collect();

        /*
            Stack the 64 images together.

            64 tensors:

                [28, 28]

            become:

                [64, 28, 28]
        */

        let images = Tensor::stack(images, 0);

        /*
            Extract labels.

            Example:

                [7, 2, 1, 9, 0, ...]

            Shape:

                [64]
        */

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [item.label as i64],
                    device,
                )
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}

// ============================================================
// TRAIN STEP
// ============================================================

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(
        &self,
        batch: MnistBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
        /*
            Forward:

                images
                  ↓
                model
                  ↓
                logits
                  ↓
            cross entropy loss
        */

        let output =
            self.forward_classification(batch.images, batch.targets);

        /*
            THIS IS BACKPROPAGATION.

            loss.backward()

            Burn walks backward through the computation graph and
            calculates partial derivatives:

                ∂L/∂w1
                ∂L/∂w2
                ∂L/∂b1
                ∂L/∂b2
                ...

            These gradients are returned.
        */

        let gradients = output.loss.backward();

        TrainOutput::new(self, gradients, output)
    }
}

// ============================================================
// VALIDATION STEP
// ============================================================

impl<B: Backend> InferenceStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> Self::Output {
        /*
            Validation does NOT need backward().

            We only run:

                forward
                ↓
                calculate loss / accuracy
        */

        self.forward_classification(batch.images, batch.targets)
    }
}

// ============================================================
// TRAINING CONFIGURATION
// ============================================================

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    pub optimizer: AdamConfig,

    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-3)]
    pub learning_rate: f64,
}

// ============================================================
// TRAIN
// ============================================================

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    println!("=====================================");
    println!("MNIST TRAINING");
    println!("=====================================");
    println!("epochs      : {}", config.num_epochs);
    println!("batch size  : {}", config.batch_size);
    println!("learning rate: {}", config.learning_rate);
    println!("=====================================");

    // --------------------------------------------------------
    // 1. Prepare output directory
    // --------------------------------------------------------

    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir)
        .expect("Could not create artifact directory.");

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Could not save config.");

    // --------------------------------------------------------
    // 2. Seed random number generator
    // --------------------------------------------------------

    B::seed(&device, config.seed);

    // --------------------------------------------------------
    // 3. Create the batcher
    // --------------------------------------------------------

    let batcher = MnistBatcher::default();

    // --------------------------------------------------------
    // 4. Training dataset
    // --------------------------------------------------------

    let train_dataset = MnistDataset::train();

    println!(
        "Training examples: {}",
        train_dataset.len()
    );

    /*
        If MNIST contains roughly:

            60,000 training images

        and:

            batch_size = 64

        then:

            60,000 / 64 ≈ 938

        optimizer steps per epoch.
    */

    let dataloader_train =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(train_dataset);

    // --------------------------------------------------------
    // 5. Test/validation dataset
    // --------------------------------------------------------

    let test_dataset = MnistDataset::test();

    println!(
        "Test examples: {}",
        test_dataset.len()
    );

    let dataloader_test =
        DataLoaderBuilder::new(batcher)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .build(test_dataset);

    // --------------------------------------------------------
    // 6. Create model
    // --------------------------------------------------------

    let model = config.model.init::<B>(&device);

    // --------------------------------------------------------
    // 7. Create optimizer
    // --------------------------------------------------------

    /*
        Adam will receive gradients produced by backward().

        Conceptually:

            gradient:

                g = ∂L/∂w

            Adam maintains:

                m = moving average of gradient
                v = moving average of squared gradient

            then updates approximately:

                w = w
                    - learning_rate *
                    m / (sqrt(v) + epsilon)
    */

    let optimizer = config.optimizer.init();

    // --------------------------------------------------------
    // 8. Training environment
    // --------------------------------------------------------

    let training = SupervisedTraining::new(
        artifact_dir,
        dataloader_train,
        dataloader_test,
    )
    .metrics((
        AccuracyMetric::new(),
        LossMetric::new(),
    ))
    .with_file_checkpointer(
        CompactRecorder::new(),
    )
    .num_epochs(config.num_epochs)
    .summary();

    // --------------------------------------------------------
    // 9. Construct learner
    // --------------------------------------------------------

    /*
        Learner connects:

            MODEL
              +
            OPTIMIZER
              +
            LEARNING RATE
    */

    let learner = Learner::new(
        model,
        optimizer,
        config.learning_rate,
    );

    // --------------------------------------------------------
    // 10. TRAIN
    // --------------------------------------------------------

    /*
        Internally Burn essentially performs:

        for epoch in 0..num_epochs {

            for batch in dataloader {

                logits = model.forward(batch.images)

                loss = cross_entropy(
                    logits,
                    batch.targets
                )

                gradients = loss.backward()

                model = optimizer.step(
                    learning_rate,
                    model,
                    gradients
                )
            }
        }
    */

    let result = training.launch(learner);

    // --------------------------------------------------------
    // 11. Save trained model
    // --------------------------------------------------------

    result
        .model
        .save_file(
            format!("{artifact_dir}/model"),
            &CompactRecorder::new(),
        )
        .expect("Could not save trained model.");

    println!();
    println!("Training complete.");
    println!("Model saved to:");
    println!("{artifact_dir}/model");
}

// ============================================================
// INFERENCE
// ============================================================

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    item: MnistItem,
) {
    println!();
    println!("=====================================");
    println!("MNIST INFERENCE");
    println!("=====================================");

    // --------------------------------------------------------
    // 1. Load training configuration
    // --------------------------------------------------------

    let config =
        TrainingConfig::load(
            format!("{artifact_dir}/config.json")
        )
        .expect("Could not load config.");

    // --------------------------------------------------------
    // 2. Load model weights
    // --------------------------------------------------------

    let record =
        CompactRecorder::new()
            .load(
                format!("{artifact_dir}/model"),
                &device,
            )
            .expect("Could not load model.");

    // --------------------------------------------------------
    // 3. Re-create model architecture
    // --------------------------------------------------------

    let model =
        config
            .model
            .init::<B>(&device)
            .load_record(record);

    // --------------------------------------------------------
    // 4. Create a single-item batch
    // --------------------------------------------------------

    let batcher = MnistBatcher::default();

    let batch =
        batcher.batch(
            vec![item.clone()],
            &device,
        );

    // --------------------------------------------------------
    // 5. Forward pass
    // --------------------------------------------------------

    let logits = model.forward(batch.images);

    println!("Raw logits:");
    println!("{logits}");

    /*
        logits shape:

            [1, 10]

        Example:

            [
                -1.1,
                 0.2,
                -3.0,
                 1.2,
                -2.2,
                 0.4,
                 7.8,
                 0.1,
                -0.8,
                 1.0
            ]

        Highest value is position 6.

        Therefore:

            prediction = digit 6
    */

    // --------------------------------------------------------
    // 6. Find largest logit
    // --------------------------------------------------------

    let prediction =
        logits.argmax(1);

    let prediction_data =
        prediction.into_data();

    let predicted_digit =
        prediction_data
            .as_slice::<i64>()
            .expect("Prediction should contain integers")[0];

    println!();
    println!("Actual digit    : {}", item.label);
    println!("Predicted digit : {}", predicted_digit);

    if predicted_digit == item.label as i64 {
        println!("Correct!");
    } else {
        println!("Incorrect.");
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    /*
        WGPU backend.

        This can run on GPU through Vulkan / Metal / DirectX /
        WebGPU depending on your platform.
    */

    type InferenceBackend = Wgpu<f32, i32>;

    /*
        Training requires automatic differentiation.

        So we wrap Wgpu:

            Wgpu

        with:

            Autodiff<Wgpu>
    */

    type TrainingBackend =
        Autodiff<InferenceBackend>;

    // --------------------------------------------------------
    // DEVICE
    // --------------------------------------------------------

    let device =
        burn::backend::wgpu::WgpuDevice::default();

    // --------------------------------------------------------
    // ARTIFACT DIRECTORY
    // --------------------------------------------------------

    let artifact_dir = "./mnist-artifacts";

    // --------------------------------------------------------
    // CONFIG
    // --------------------------------------------------------

    let config =
        TrainingConfig::new(
            ModelConfig::new(),
            AdamConfig::new(),
        )
        .with_num_epochs(5)
        .with_batch_size(64)
        .with_learning_rate(1e-3);

    // --------------------------------------------------------
    // TRAIN
    // --------------------------------------------------------

    train::<TrainingBackend>(
        artifact_dir,
        config,
        device.clone(),
    );

    // --------------------------------------------------------
    // SELECT AN IMAGE FOR INFERENCE
    // --------------------------------------------------------

    let test_dataset =
        MnistDataset::test();

    let item =
        test_dataset
            .get(42)
            .expect("MNIST item should exist.");

    // --------------------------------------------------------
    // INFERENCE
    // --------------------------------------------------------

    infer::<InferenceBackend>(
        artifact_dir,
        device,
        item,
    );
}
