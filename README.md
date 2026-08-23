# WheatSeedling-LA-Code

Minimal reproducibility code for **WheatSeedling-LA v1.0**, a near-ground, low-angle field image dataset for wheat seedling detection and image-level counting.

## Dataset

- Dataset: WheatSeedling-LA v1.0
- Zenodo DOI: https://doi.org/10.5281/zenodo.21847420
- Code repository: https://github.com/bin365-hub/WheatSeedling-LA-Code
- Class: `0 = Wheat`
- Original field images: 2,203
- Original annotated instances: 26,613
- Predefined subsets: 1,538 train / 443 validation / 222 test
- Test instances: 2,650
- Augmented training resource: 7,690 images / 93,025 instances
- Public-release images: original aspect ratio preserved, longer side limited to 1,920 pixels, JPEG quality 92

The predefined subsets are intended primarily for benchmark reproduction and should not be interpreted as a strict cross-site or strict acquisition-sequence partition.

This repository contains code and configuration files only. **Trained benchmark model weights are not distributed.**

## Repository structure

```text
WheatSeedling-LA-Code/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── configs/
│   ├── dataset_original.yaml
│   └── dataset_augmented.yaml
├── scripts/
│   ├── validate_dataset.py
│   ├── run_baseline.py
│   └── annotation_consistency.py
└── docs/
    └── REPRODUCIBILITY.md
```

## Installation

Reference software environment used in the manuscript:

```text
Python       3.8.20
PyTorch      2.1.0+cu121
Ultralytics  8.4.104
CUDA         12.1
GPU          NVIDIA GeForce RTX 4060 Laptop GPU
```

Install the repository dependencies with:

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your local CUDA/CPU environment separately.

## Dataset location

A convenient layout is:

```text
workspace/
├── WheatSeedling-LA-Code/
└── WheatSeedling-LA_v1.0/
```

The supplied YAML files assume the dataset is a sibling directory of this repository. Edit the `path:` entry if your local location differs.

## 1. Validate the dataset and reproduce Table 4 statistics

```bash
python scripts/validate_dataset.py \
  --dataset-root ../WheatSeedling-LA_v1.0 \
  --output-dir results/table4
```

The script checks:

- image-label correspondence;
- YOLO field count, class ID and coordinate validity;
- expected split and instance counts;
- readability of final public-release JPEG files;
- maximum released-image longer side of 1,920 pixels;
- exact SHA-256 duplicates across the predefined original subsets;
- augmented training-set image/label and instance counts.

It also calculates:

- mean number of instances per image;
- median normalized bounding-box area;
- median bounding-box aspect ratio;
- median brightness;
- median contrast;
- median sharpness (variance of the Laplacian response).

Current final-public-release Table 4 values:

| Metric | Training | Validation | Test |
|---|---:|---:|---:|
| Mean instances/image | 12.10 | 12.09 | 11.94 |
| Median normalized bbox area | 0.0138 | 0.0132 | 0.0143 |
| Median bbox aspect ratio | 0.307 | 0.297 | 0.286 |
| Median brightness | 136.93 | 137.82 | 137.96 |
| Median contrast | 42.93 | 42.88 | 40.81 |
| Median sharpness | 733.41 | 766.11 | 748.45 |

## 2. Baseline training and evaluation

All baseline functionality is combined in `scripts/run_baseline.py`.

Reference training settings:

```text
imgsz        = 640
batch        = 16
epochs       = 300
patience     = 50
optimizer    = AdamW
lr0          = 0.002
beta1        = 0.9
beta2        = 0.999
weight decay = 0.0005
seed         = 42
deterministic= True
AMP          = True
workers      = 2
```

### Train

YOLO11n with the original training subset:

```bash
python scripts/run_baseline.py train \
  --model yolo11n.pt \
  --data configs/dataset_original.yaml \
  --name yolo11n_original
```

YOLO26n with the augmented training resource:

```bash
python scripts/run_baseline.py train \
  --model yolo26n.pt \
  --data configs/dataset_augmented.yaml \
  --name yolo26n_augmented
```

### Evaluate detection

```bash
python scripts/run_baseline.py detection \
  --weights path/to/best.pt \
  --data configs/dataset_original.yaml \
  --output results/detection.json
```

The script reports precision, recall, mAP@0.5, and mAP@0.5:0.95 on the predefined test subset.

### Select the counting threshold on validation only

```bash
python scripts/run_baseline.py threshold \
  --weights path/to/best.pt \
  --dataset-root ../WheatSeedling-LA_v1.0 \
  --split val \
  --output results/threshold_search.csv
```

The search grid is 0.01–0.99 with a step of 0.01. The threshold with the minimum validation MAE is selected.

Reference thresholds reported before any later baseline revalidation are:

| Training condition | YOLO11n | YOLO26n |
|---|---:|---:|
| Original | 0.42 | 0.33 |
| Augmented | 0.42 | 0.36 |

### Evaluate counting

```bash
python scripts/run_baseline.py counting \
  --weights path/to/best.pt \
  --dataset-root ../WheatSeedling-LA_v1.0 \
  --split test \
  --conf 0.42 \
  --output results/counting.csv
```

Reported metrics include MAE, RMSE, R², mean signed error (`prediction - ground truth`), percentage of images with absolute error ≤ 2 seedlings, predicted total, and ground-truth total.

## 3. Annotation-consistency calculation

For two YOLO annotation directories describing the same images:

```bash
python scripts/annotation_consistency.py \
  --reference path/to/consensus_labels \
  --comparison path/to/reannotation_labels \
  --iou-threshold 0.5
```

The script performs one-to-one Hungarian matching and reports automatic matched pairs, instance-level F1, and matched-box IoU statistics.

Manual adjudication is a separate human-review step and is not automated by this script.

## Citation

If you use the dataset, cite:

```text
WheatSeedling-LA v1.0. Zenodo.
https://doi.org/10.5281/zenodo.21847420
```

Please also cite the accompanying data descriptor once it is published.

For this code repository, see `CITATION.cff`.

## Licenses

- Code in this repository: MIT License.
- WheatSeedling-LA dataset: CC BY 4.0.

## Notes

- Dataset images are not redistributed in this repository.
- Trained benchmark model weights are not distributed.
- Do not commit local datasets, model checkpoints, or experiment output directories.
