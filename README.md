
# PAF-KIP-OSTTA

Official PyTorch implementation of **"Stabilizing Open-Set Test-Time Adaptation via Primary-Auxiliary Filtering and Knowledge-Integrated Prediction"** (BMVC 2025).

---

## 1. Environment Setup

**Tested with:** Python 3.9, PyTorch 1.12~2.8, CUDA 11.6+, NVIDIA GPU with BF16 support (RTX 3090, A100, etc.)

```bash
# 1) Create conda environment
conda create -n pafkip python=3.9 -y
conda activate pafkip

# 2) Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` installation fails, install manually:
```bash
pip install torch torchvision robustbench prettytable tqdm iopath scikit-learn scipy termcolor wandb
pip install git+https://github.com/fra31/auto-attack
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
```

---

## 2. Dataset Preparation

All datasets should be placed under the home directory (`~/`) by default.
You can change the location with `--data_dir`.

### 2.1 Closed-Set Datasets (ID)

These are required. robustbench will auto-download them if not found.

#### CIFAR-10-C
```bash
cd ~
wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar
# Result: ~/CIFAR-10-C/  (contains gaussian_noise.npy, labels.npy, ...)
```

#### CIFAR-100-C
```bash
cd ~
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar
# Result: ~/CIFAR-100-C/  (contains gaussian_noise.npy, labels.npy, ...)
```

#### ImageNet-C
```bash
cd ~/imagenet
# Download all 15 corruption types from:
# https://zenodo.org/records/2235448
# Each corruption type is a separate .tar file (blur.tar, digital.tar, noise.tar, weather.tar, extra.tar)
wget https://zenodo.org/records/2235448/files/blur.tar
wget https://zenodo.org/records/2235448/files/digital.tar
wget https://zenodo.org/records/2235448/files/noise.tar
wget https://zenodo.org/records/2235448/files/weather.tar
wget https://zenodo.org/records/2235448/files/extra.tar
for f in blur.tar digital.tar noise.tar weather.tar extra.tar; do tar -xvf $f; done
# Result: ~/imagenet/ImageNet-C/{corruption_type}/{severity}/  (ImageFolder structure)
```

### 2.2 Open-Set Datasets (OOD)

You need **at least one** OOD dataset for open-set TTA experiments.
The easiest option is `--ood_dataset gaussian` which requires **no download**.

#### Option A: Synthetic Noise (No Download Needed)
```bash
# Just use --ood_dataset gaussian or --ood_dataset uniform
# Gaussian/uniform noise images are generated on-the-fly
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian
```

#### Option B: SVHN-C (Default OOD for CIFAR)

SVHN-C contains corrupted SVHN digit images as `.npy` files.

```bash
cd ~
# Download SVHN-C (corrupted SVHN dataset)
# Expected structure:
# ~/SVHN-C/
#   labels.npy
#   gaussian_noise.npy
#   shot_noise.npy
#   impulse_noise.npy
#   defocus_blur.npy
#   glass_blur.npy
#   motion_blur.npy
#   zoom_blur.npy
#   snow.npy
#   frost.npy
#   fog.npy
#   brightness.npy
#   contrast.npy
#   elastic_transform.npy
#   pixelate.npy
#   jpeg_compression.npy
```

> **Note:** SVHN-C is not publicly available on a single download link.
> You can generate it by applying the 15 standard corruptions to the SVHN test set,
> or use `--ood_dataset tiny_imagenet` or `--ood_dataset gaussian` as alternatives.

#### Option C: Tiny-ImageNet-C

```bash
cd ~
# Download from: https://zenodo.org/records/2536630
wget https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar
tar -xvf Tiny-ImageNet-C.tar
# Result: ~/Tiny-ImageNet-C/{corruption_type}/{severity}/{class_id}/*.JPEG  (ImageFolder structure)
```

#### Option D: Places365-C / Textures-C (CIFAR)

```bash
# Places365-C
# Expected: ~/PLACES365-C/{corruption_type}/{severity}/{class_id}/*.jpg

# Textures-C (for CIFAR: 32x32 resized)
# Expected: ~/Texture-C/{corruption_type}/{severity}/{class_id}/*.jpg
```

#### Option E: ImageNet-O-C (Default OOD for ImageNet)

```bash
# Expected: ~/imagenet/ImageNet-O-C/{corruption_type}/{severity}/{class_id}/*.JPEG
```

### 2.3 Expected Directory Structure Summary

```
~/
├── CIFAR-10-C/                    # CIFAR-10 corrupted (required for cifar10 experiments)
│   ├── labels.npy
│   ├── gaussian_noise.npy
│   ├── shot_noise.npy
│   └── ...
├── CIFAR-100-C/                   # CIFAR-100 corrupted (required for cifar100 experiments)
│   ├── labels.npy
│   └── ...
├── SVHN-C/                        # OOD for CIFAR (default, or use --ood_dataset alternatives)
│   ├── labels.npy
│   └── ...
├── Tiny-ImageNet-C/               # OOD for CIFAR (alternative)
│   └── gaussian_noise/5/{class_id}/*.JPEG
├── PLACES365-C/                   # OOD for CIFAR (optional)
│   └── gaussian_noise/5/{class_id}/*.jpg
├── Texture-C/                     # OOD for CIFAR (optional)
│   └── gaussian_noise/5/{class_id}/*.jpg
└── imagenet/
    ├── ImageNet-C/                # ImageNet corrupted (required for imagenet experiments)
    │   └── gaussian_noise/5/{class_id}/*.JPEG
    ├── ImageNet-O-C/              # OOD for ImageNet
    │   └── gaussian_noise/5/{class_id}/*.JPEG
    ├── PLACES365-C/               # OOD for ImageNet (optional)
    └── Textures-C/                # OOD for ImageNet (optional)
```

### 2.4 Model Checkpoints

- **CIFAR-10:** Included in repo at `cifar/ckpt/cifar10/corruptions/Hendrycks2020AugMix_WRN.pt`
- **CIFAR-100:** Auto-downloaded by robustbench on first run (requires internet)
- **ImageNet:** Auto-downloaded by robustbench on first run (requires internet)

---

## 3. Quick Start

The simplest way to run PAF-KIP **without downloading any extra datasets**:

```bash
cd cifar

# CIFAR-10 + Gaussian noise as OOD (no extra data needed)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian

# Closed-set only (no OOD data needed)
python main.py --adaptation ours --dataset cifar10 --open_set_tta False
```

If you have Tiny-ImageNet-C downloaded:
```bash
python main.py --adaptation ours --dataset cifar10 --ood_dataset tiny_imagenet
```

---

## 4. Reproducing Paper Results

### Table 1: CIFAR-10-C (Paper Section 3.2)

Settings: WideResNet-40-2, batch_size=200, Adam, lr=0.001, alpha=2.0, mt=0.999, thr=0.4

```bash
cd cifar

# CIFAR-10-C + SVHN-C (Table 1, column 1)
python main.py --adaptation ours --dataset cifar10 --ood_dataset svhn

# CIFAR-10-C + Tiny-ImageNet-C (Table 1, column 2)
python main.py --adaptation ours --dataset cifar10 --ood_dataset tiny_imagenet

# CIFAR-10-C + Places365-C (Table 1, column 3)
python main.py --adaptation ours --dataset cifar10 --ood_dataset places365

# CIFAR-10-C + Textures-C (Table 1, column 4)
python main.py --adaptation ours --dataset cifar10 --ood_dataset textures
```

Expected results (averaged over 15 corruptions):
| OOD Dataset | ACC | AUROC | H-Score |
|-------------|-----|-------|---------|
| SVHN-C | 87.49% | 97.66% | 92.30% |
| TinyImageNet-C | 88.26% | 90.61% | 89.42% |
| Places365-C | 88.35% | 94.24% | 91.20% |
| Textures-C | 87.80% | 97.71% | 92.49% |

### Table 2: CIFAR-100-C (Paper Section 3.2)

```bash
cd cifar

# CIFAR-100-C + SVHN-C
python main.py --adaptation ours --dataset cifar100 --ood_dataset svhn

# CIFAR-100-C + Tiny-ImageNet-C
python main.py --adaptation ours --dataset cifar100 --ood_dataset tiny_imagenet
```

Expected results:
| OOD Dataset | ACC | AUROC | H-Score |
|-------------|-----|-------|---------|
| SVHN-C | 62.59% | 97.61% | 76.27% |
| TinyImageNet-C | 63.79% | 84.16% | 72.57% |

### Table 3: ImageNet-C (Paper Section 3.2)

Settings: ResNet-50, batch_size=64, SGD, lr=0.00025, ours_alpha=0.7, mt=0.999, thr=0.4

```bash
cd imagenet

# ImageNet-C + Places365-C
python main.py --adaptation ours --ours_alpha 0.7 --lr 0.00025 --batch_size 64

# ImageNet-C + Textures-C
# (requires modifying the OOD dataset loading in main.py)
```

Expected results:
| OOD Dataset | ACC | AUROC | H-Score |
|-------------|-----|-------|---------|
| Places365-C | 48.22% | 84.18% | 61.32% |
| Textures-C | 47.96% | 82.91% | 60.77% |

### Running Baseline Methods

```bash
cd cifar

# Source (no adaptation)
python main.py --adaptation source --dataset cifar10 --ood_dataset tiny_imagenet

# Tent
python main.py --adaptation tent --dataset cifar10 --ood_dataset tiny_imagenet

# CoTTA
python main.py --adaptation cotta --dataset cifar10 --ood_dataset tiny_imagenet

# EATA
python main.py --adaptation eata --dataset cifar10 --ood_dataset tiny_imagenet

# OSTTA
python main.py --adaptation ostta --dataset cifar10 --ood_dataset tiny_imagenet

# RoTTA
python main.py --adaptation rotta --dataset cifar10 --ood_dataset tiny_imagenet

# SoTTA
python main.py --adaptation sotta --dataset cifar10 --ood_dataset tiny_imagenet

# STAMP
python main.py --adaptation stamp --dataset cifar10 --ood_dataset tiny_imagenet
```

---

## 5. All Arguments Reference

### CIFAR (`cifar/main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptation` | `tent` | Method: `source`, `tent`, `cotta`, `eata`, `ostta`, `sotta`, `stamp`, `rotta`, `ours` |
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--ood_dataset` | `svhn` | `svhn`, `tiny_imagenet`, `places365`, `textures`, `gaussian`, `uniform` |
| `--open_set_tta` | `True` | `True` for open-set, `False` for closed-set only |
| `--open_set_ratio` | `1.0` | Ratio of OOD to ID samples |
| `--severity` | `5` | Corruption severity (1-5) |
| `--num_ex` | `10000` | Number of test examples per corruption |
| `--batch_size` | `200` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--method` | `Adam` | Optimizer: `Adam`, `SGD`, `SAM` |
| `--alpha` | `2.0` | OOD loss weight (alpha in paper) |
| `--mt` | `0.999` | EMA decay rate (beta in paper) |
| `--thr` | `0.4` | Entropy threshold ratio (tau = thr * log(C)) |
| `--n_aug` | `1` | Number of augmentations for EMA model |
| `--data_dir` | `~` | Root directory for datasets |
| `--ckpt_dir` | `./ckpt` | Checkpoint directory |
| `--rng_seed` | `1` | Random seed |
| `--continual` | `True` | `True` for continual adaptation (no reset between domains) |

### ImageNet (`imagenet/main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptation` | `tent` | Method: `source`, `norm`, `tent`, `cotta`, `eata`, `ostta`, `caftta`, `ours` |
| `--arch` | `Hendrycks2020AugMix` | Model architecture (ResNet-50) |
| `--ours_alpha` | `2.0` | OOD loss weight (**use 0.7 for paper results**) |
| `--batch_size` | `32` | Batch size (**use 64 for paper results**) |
| `--lr` | `0.001` | Learning rate (**use 0.00025 for paper results**) |
| `--method` | `SGD` | Optimizer: `Adam`, `SGD` |
| `--mt` | `0.999` | EMA decay rate |
| `--thr` | `0.4` | Entropy threshold ratio |
| `--n_aug` | `1` | Number of augmentations |
| `--data_dir` | `~/imagenet` | Root directory for ImageNet datasets |
| `--num_ex` | `5000` | Number of test examples per corruption |

---

## 6. Method Overview

**PAF (Primary-Auxiliary Filtering):** Entropy-based dual-filter sample categorization
- Primary filter (adapting model): Captures current domain knowledge
- Auxiliary filter (EMA model): Provides stability and prevents error accumulation
- ID samples (both filters say low entropy): Soft-weighted entropy minimization
- OOD samples (both filters say high entropy): Hard entropy maximization

**KIP (Knowledge-Integrated Prediction):** Confidence-based weighted ensemble
- Combines logits from: adapting model, EMA model, and frozen source model
- Per-sample weights based on each model's confidence relative to the mean

---

## 7. Quantization (MX-FP4 / MX-INT4 + W4A4 PTQ stack)

A self-contained quantization toolkit lives in [`quantization/`](quantization/).
Two layers, both wired through `cifar/main.py` flags:

- **`quantization.mx`** — OCP **Microscaling** fake-quant (MX-FP4 E2M1 / MX-INT4),
  group-shared E8M0 power-of-2 scale along the convolution contraction vector
  (`C_in * k_h * k_w` via `F.unfold`). Each `MXQuantConv2d` exposes
  `weight_fake_quant` and `act_fake_quant` as named `nn.Module`s, so
  `torch.fx.symbolic_trace` produces a graph with explicit quantize/dequantize
  `call_module` nodes — matching the torch.ao `prepare_fx` pattern (lowerable
  to a real MX-INT4 / MX-FP4 NPU backend).
- **`quantization.ptq`** — hand-rolled W4A4 stack: AdaRound, BRECQ, QDrop,
  NIPQ (learnable scale + mixed-precision bit allocation), SmoothQuant.

### 7.1 Quick start — MX (Microscaling)

`BF16` baseline = **78.37 %** mean ACC on CIFAR-10-C / PAF-KIP / gaussian OOD.
Within 2 % is the deployable target; commands below are exactly the configs we measured.

```bash
cd cifar

# Best deployable: MX-INT4 with group size 16 (skip first conv; HW standard)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --mx_quantize --mx_format int4 --mx_group_size 16

# OCP-spec group size (32)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --mx_quantize --mx_format int4 --mx_group_size 32

# MX-FP4 (E2M1) variant
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --mx_quantize --mx_format fp4 --mx_group_size 32

# Strict HW-faithful: quantize the first Conv2d too (no exceptions)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --mx_quantize --mx_format int4 --mx_group_size 32 --mx_no_skip_first
```

Measured numbers (`--rng_seed 1`, batch_size 200, 15 corruptions, severity 5):

| Config | Mean ACC | Δ vs BF16 | AUROC | Within 2 %? |
|---|---|---|---|---|
| **MX-INT4 g=16** | **77.73** | **−0.64** | 99.75 | ✅ |
| MX-INT4 g=32 | 77.50 | −0.87 | 99.75 | ✅ |
| MX-FP4 g=16 | 76.77 | −1.60 | 99.73 | ✅ |
| MX-FP4 g=32 | 76.80 | −1.57 | 99.76 | ✅ |
| MX-INT4 g=32, no-skip-first (strict HW) | 76.72 | −1.65 | 99.75 | ✅ |
| MX-FP4 g=32, no-skip-first (strict HW)  | 75.33 | −3.04 | 99.69 | ⚠️ |

Observations:
- **AUROC stays essentially identical** to BF16 (≤0.05 pp loss) — OOD detection is not affected.
- **MX-INT4 outperforms MX-FP4** by ~1 pp on this CNN/TTA setup: E2M1's level grid is sparse in the 4–6 region, but BN-normalized CNN activations populate that range densely (the trade-off is reversed for LLM activations).
- **First-conv skip costs ~0.1 pp on INT4, ~1.5 pp on FP4** — keeping `conv1` in FP/INT8 is the common HW practice and what `--mx_skip_first` (default) does.

### 7.2 Hand-rolled W4A4 PTQ stack

```bash
cd cifar

# Per-tensor dynamic W4A4 (basic deployable)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --quantize --w_bits 4 --a_bits 4 --act_percentile 0.999 --mse_weight --skip_first_conv

# Best non-deployable accuracy ceiling: per-channel activation + BRECQ + QDrop
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --quantize --w_bits 4 --a_bits 4 \
  --act_percentile 0.999 --mse_weight --skip_first_conv \
  --brecq --adaround_samples 512 --brecq_iters 10000 \
  --act_granularity per_channel --qdrop_p 0.5

# NIPQ — learnable per-tensor scale + mixed-precision A4/A8 (fully deployable)
python main.py --adaptation ours --dataset cifar10 --ood_dataset gaussian \
  --quantize --w_bits 4 --a_bits 4 --mse_weight --skip_first_conv \
  --nipq --nipq_target_bit 4.5 --nipq_iters 2000
```

### 7.3 CLI flags reference

| Flag | Meaning |
|---|---|
| `--mx_quantize` | Enable MX (Microscaling) fake-quant for all Conv2d |
| `--mx_format {fp4,int4}` | Element format. `fp4` = E2M1, `int4` = signed INT4 |
| `--mx_group_size 16` or `32` | Block size along contraction dim. OCP spec = 32 |
| `--mx_no_e8m0` | Use float per-group scale instead of E8M0 (debug only) |
| `--mx_no_act` | Quantize weight only (skip activation MX quant) |
| `--mx_no_skip_first` | Quantize the first Conv2d too (strict HW-faithful) |
| `--quantize` | Enable hand-rolled W4A4 PTQ |
| `--adaround` / `--brecq` | Calibration: per-layer AdaRound or block-wise BRECQ |
| `--qdrop_p 0.5` | QDrop probability during calibration |
| `--act_granularity {per_tensor,per_channel,per_sample}` | Activation grouping |
| `--act_percentile 0.999` | Percentile clip for dynamic activation scale |
| `--mse_weight` | MSE-optimal weight scale (BRECQ pre-step) |
| `--skip_first_conv` | Keep the first conv in FP |
| `--nipq` | NIPQ learnable scale + mixed-precision bit allocation |
| `--nipq_target_bit 4.5` | MAC-weighted target avg activation bit (NIPQ) |
| `--smoothquant --sq_alpha 0.5` | SmoothQuant BN→conv migration |

### 7.4 torch.fx integration

```python
from quantization import apply_mx_quantization, fx_trace

model = apply_mx_quantization(model, fp4=False, group_size=16)   # MX-INT4 g=16
gm    = fx_trace(model.block1.layer.0)                           # any MXQuantConv2d
for n in gm.graph.nodes:
    print(n.op, n.target, '→', n.name)
# call_module      weight_fake_quant      → weight_fake_quant
# call_module      act_fake_quant         → act_fake_quant
# call_function    matmul                 → matmul
# ...
```

The `call_module` nodes are exactly what `torch.ao.quantization.prepare_fx`
yields with QuantStub / DeQuantStub — swap them for HW kernels to deploy
on MX-FP4 / MX-INT4 silicon (Blackwell, MX-capable NPUs).

See [`quantization/README.md`](quantization/README.md) for full API reference.

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'autoattack'` | `pip install git+https://github.com/fra31/auto-attack` |
| `ModuleNotFoundError: No module named 'wandb'` | `pip install wandb` |
| `ModuleNotFoundError: No module named 'termcolor'` | `pip install termcolor` |
| `FileNotFoundError: .../SVHN-C/labels.npy` | SVHN-C not found. Use `--ood_dataset gaussian` or `--ood_dataset tiny_imagenet` instead |
| `CUDA out of memory` | Reduce `--batch_size` (e.g., 100 for CIFAR, 16 for ImageNet) |
| `BF16 not supported` | Requires GPU with compute capability >= 8.0 (Ampere+: RTX 3090, A100, etc.) |
| CIFAR-100 checkpoint missing | robustbench auto-downloads on first run (requires internet) |
| Low accuracy compared to paper | Ensure `--continual True` (no reset) and correct `--alpha`/`--ours_alpha` values |

---

## 9. Citation

```bibtex
@inproceedings{lee2025pafkip,
  title={Stabilizing Open-Set Test-Time Adaptation via Primary-Auxiliary Filtering and Knowledge-Integrated Prediction},
  author={Lee, Byung-Joon and Lee, Jin-Seop and Lee, Jee-Hyong},
  booktitle={BMVC},
  year={2025}
}
```
