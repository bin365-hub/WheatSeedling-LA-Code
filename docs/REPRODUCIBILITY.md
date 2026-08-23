# Reproducibility

## Dataset

WheatSeedling-LA v1.0 contains:

- 2,203 original field images;
- 26,613 original wheat seedling instances;
- 1,538 / 443 / 222 predefined train / validation / test images;
- 2,650 test instances;
- 7,690 images and 93,025 instances in the augmented training resource.

The final public-release JPEG files preserve the original aspect ratio, have a
maximum longer side of 1,920 pixels, and were re-encoded at JPEG quality 92.

The predefined subsets are intended for benchmark reproduction and are not a
strict cross-site or strict acquisition-sequence partition.

## Reference environment

- Python 3.8.20
- PyTorch 2.1.0+cu121
- Ultralytics 8.4.104
- CUDA 12.1
- NVIDIA GeForce RTX 4060 Laptop GPU

## Reference training settings

- `imgsz=640`
- `batch=16`
- `epochs=300`
- `patience=50`
- `optimizer=AdamW`
- `lr0=0.002`
- β1 = 0.9
- β2 = 0.999
- weight decay = 0.0005
- seed = 42
- deterministic mode enabled
- AMP enabled
- workers = 2

Each reference configuration was trained once with the fixed random seed.

## Counting evaluation

The counting confidence threshold must be selected on the validation subset
only. The search grid is 0.01–0.99 in increments of 0.01. The threshold with
minimum validation MAE is then fixed for test-set counting evaluation.

`run_baseline.py counting` reports:

- MAE
- RMSE
- R²
- mean signed error = mean(predicted count - ground-truth count)
- percentage of images with absolute error <= 2 seedlings
- predicted total
- ground-truth total

## Annotation consistency

`annotation_consistency.py` reproduces the automatic Hungarian matching stage
at a user-defined IoU threshold. Manual adjudication remains a human-review
procedure and is not automated.

## Model weights

Trained benchmark model weights are not distributed.
