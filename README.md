# ALMRI — Active Learning for Multi-organ MRI Segmentation

Active learning pipeline for segmenting abdominal MRI volumes (left kidney, right kidney, pancreas) using **DynUNet** (MONAI DeepEdit) and **SAM/MedSAM** baselines. The project supports incremental subject acquisition, checkpoint-based finetuning across growing training sets, and evaluation with per-organ Dice scores.

## Overview

The workflow has three main stages:

1. **Preprocessing** — Extract 2D slices and SAM image embeddings from 3D NIfTI volumes (`pre_embeddings.py`).
2. **Training** — Active learning or finetuning with DynUNet (`active_dynunet.py`), or SAM fine-tuning on embeddings (`train.py`).
3. **Evaluation** — Inference and Dice computation (`eval_dynunet.py`, `eval.py`).

```
datasets/          NIfTI images and labels (not tracked in git)
pre_embeddings.py  Slice extraction + SAM embeddings → .npz pools
active_dynunet.py  DynUNet active learning / finetune
eval_dynunet.py    DynUNet inference and Dice evaluation
train.py / eval.py SAM-based active learning baseline
```

## Segmentation targets

| Label | ID |
|-------|----|
| Background | 0 |
| Left kidney | 1 |
| Right kidney | 2 |
| Pancreas | 3 |

DynUNet uses a 5-channel input (1 intensity + 4 guidance channels) via MONAI DeepEdit transforms.

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate medsam
```

Key dependencies: Python 3.10, PyTorch 2.0, MONAI 0.9, SimpleITK, nibabel, scikit-image.

On HPC clusters, load CUDA before activating the environment:

```bash
module load cuda
source activate medsam
```

Alternatively, use `miniconda_setup.sh` to install Miniconda from scratch.

### Data layout

Expected directory structure (paths are configurable via CLI flags):

```
ALMRI/
├── datasets/
│   ├── RAINE_organ_51/          # Full dataset prefix (SAM pipeline)
│   ├── trainingset_45/          # 45 subjects
│   │   └── labels/final/*.nii.gz
│   ├── trainingset_49/          # 45 + 4 new subjects
│   │   └── labels/final/
│   └── trainingset_51/          # 49 + 2 new subjects
│       └── labels/final/
├── checkpoints/                 # Model weights (gitignored)
└── results/                     # Loss curves, Dice, sampling logs (gitignored)
```

Image files use the naming convention `sub-XXXXX.FatImaging_W.nii.gz`.

## Training (DynUNet)

`active_dynunet.py` provides three subcommands for the DynUNet training pipeline.

### Active learning on trainingset_45

Adds subjects one at a time using **random** or **epistemic entropy** sampling (`strategy.py`).

| Setting | Value |
|---------|-------|
| Bootstrap | First 5 subjects added sequentially (entropy mode only) |
| Training start | 5 subjects |
| Epochs at 5 subjects | 100 |
| Epochs per new subject | 30 |
| Milestone checkpoints | 35, 40, 45 subjects |

```bash
# Entropy-based active learning
python active_dynunet.py active \
    --strategy entropy \
    --spatial_size 192 192 96 \
    --num_mc_samples 5 \
    --initial_subjects 5 \
    --initial_epochs 100 \
    --incremental_epochs 30 \
    --seed 2025

# Random baseline
python active_dynunet.py active \
    --strategy random \
    --spatial_size 192 192 96 \
    --seed 2024
```

Checkpoints are saved under:

```
checkpoints/DynUNet_trainingset45_{random,entropy}_active/seed{SEED}/init100_inc30/
```

Acquisition order and entropy scores are logged to `results/sampling/trainingset45_active/{strategy}/`.

### Finetune onto trainingset_49

Loads the 45-subject entropy checkpoint and adds four subjects in fixed order: `sub-41810`, `sub-16880`, `sub-12400`, `sub-14870`.

```bash
python active_dynunet.py finetune49 --seed 2025
```

Milestone checkpoints at 46, 47, 48, 49 subjects.

### Finetune onto trainingset_51

Loads the 49-subject checkpoint and adds two subjects: `sub-52220`, `sub-50710`.

```bash
python active_dynunet.py finetune51 --seed 2025
```

Milestone checkpoints at 50 and 51 subjects.

## Evaluation (DynUNet)

`eval_dynunet.py` supports two modes.

### `infer` — single-checkpoint inference

Runs one checkpoint on a directory of NIfTI images. Computes per-organ and mean Dice when labels are provided. Saves label maps (`.nii.gz`) and one-hot masks (`.npz`).

```bash
python eval_dynunet.py infer \
    --spatial_size 192 192 96 \
    --checkpoint checkpoints/DynUNet_trainingset45_entropy_active/seed2025/init100_inc30/DynUNet_45subjects_seed2025.pt \
    --image_dir datasets/trainingset_51 \
    --label_dir datasets/trainingset_51/labels/final \
    --pred_dir inference_outputs/pred_trainingset_51 \
    --dice_dir results/dice/trainingset_51
```

Outputs in `--dice_dir`: `dice.json`, `dice_per_case.npy`, `dice_mean_per_organ.npy`.

### `curve` — checkpoint

Evaluates a series of active-learning checkpoints (`*latest.pth`) and plots Dice vs. training pool size for a single organ.

```bash
python eval_dynunet.py curve \
    --spatial_size 192 192 96 \
    --task MRI_LeftKidney \
    --label_id 1 \
    --num_epochs 100 \
    --seed 2024 \
    --mode test
```

## SAM / MedSAM baseline

The SAM pipeline operates on precomputed `.npz` embedding pools.

### Generate embeddings

```bash
python pre_embeddings.py \
    --task medsam_MRI_LeftKidney_sagittal_norm \
    --label_id 1 \
    --mode test \
    --view 2 \
    --prefix datasets/RAINE_organ_51 \
    --checkpoint checkpoints/MedSAM/medsam_vit_b.pth \
    --image_size 256
```

### Train and evaluate SAM

```bash
python train.py --task MRI_LeftKidney --label_id 1 --base_model SAM_roi --seed 2024
python eval.py  --task MRI_LeftKidney --label_id 1 --num_epochs 20 --seed 2024
```

ROI-guided variants are in `eval_roi.py`.

## SLURM jobs

Submit from the project root:

```bash
cd /path/to/labelling/ALMRI
sbatch experiments/run_train_dyn.sh      # DynUNet active learning (random, seed 2024)
sbatch experiments/run_eval_dyn.sh       # DynUNet inference on trainingset_51
sbatch experiments/run_train.sh          # SAM active learning
sbatch experiments/run_eval.sh           # SAM evaluation
sbatch experiments/run_embeddings.sh     # MedSAM embedding extraction
```

Edit `#SBATCH --account`, partition, and QoS settings for your cluster before submitting.

## Project structure

```
ALMRI/
├── active_dynunet.py       DynUNet training (active / finetune49 / finetune51)
├── eval_dynunet.py         DynUNet evaluation (curve / infer)
├── strategy.py             Sampling: random, sequential, epistemic entropy
├── metrics.py              Dice coefficient
├── utils.py                Datasets, NIfTI I/O, normalization helpers
├── pre_embeddings.py       SAM slice + embedding preprocessing
├── train.py                SAM active-learning fine-tuning
├── eval.py                 SAM batch inference
├── eval_roi.py             SAM inference with ROI bounding boxes
├── segment_anything/       SAM model code
├── experiments/            SLURM launch scripts
├── notebooks/              Analysis and dataset preparation
│   ├── prepare_dataset.ipynb
│   ├── research_questions.ipynb
│   └── ploting.ipynb
├── environment.yml         Conda environment
└── miniconda_setup.sh      Miniconda installer
```

## Active learning strategies

Defined in `strategy.py`:

| Strategy | Description |
|----------|-------------|
| `random_sampling` | Uniform random choice from the unlabeled pool |
| `sequential_sampling` | Lexicographic order (entropy bootstrap only) |
| `entropy_sampling` | MONAI Label–style epistemic entropy via MC dropout; selects highest-uncertainty subject |

## Outputs

| Path | Contents |
|------|----------|
| `checkpoints/` | DynUNet `.pt` and SAM `.pth` weights |
| `results/loss/` | Per-step training loss `.npy` files |
| `results/sampling/` | Acquisition order and entropy score JSON |
| `results/dice/` | Evaluation Dice summaries |
| `inference_outputs/` | Predicted label maps and one-hot `.npz` |
| `figures/` | Loss and Dice curves (legacy curve mode) |
| `results/logs/` | SLURM stdout/stderr |

These directories are gitignored; create them before running jobs (`mkdir -p results/logs`).

## Notes

- DynUNet training uses `DataLoader(shuffle=False)` to preserve reproducible subject order within each pool.
- Entropy and random active-learning runs should be kept as **separate experiments** (do not mix strategies in one run).
- The finetune stages use a **fixed subject order** rather than active sampling.
- Optional dependency: `monailabel` for native epistemic entropy scoring; a local fallback is implemented in `strategy.py` if it is not installed.
