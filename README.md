
# PAF-KIP-OSTTA
This is official PyTorch implementation of "Stabilizing Open-Set Test-Time Adaptation via Primary-Auxiliary Filtering and Knowledge-Integrated Prediction" (BMVC 2025).

## Requirements

```bash
pip install torch torchvision robustbench prettytable tqdm iopath scikit-learn scipy
```

## Dataset Preparation

### CIFAR-10-C / CIFAR-100-C
Download and extract to `~/CIFAR-10-C` and `~/CIFAR-100-C`:
- CIFAR-10-C: https://zenodo.org/record/2535967
- CIFAR-100-C: https://zenodo.org/record/3555552

### ImageNet-C / ImageNet-O-C
Download and extract to `~/imagenet/`:
- ImageNet-C
- ImageNet-O-C (for OOD samples)

## Usage

### CIFAR-10/100 (PAF-KIP)

```bash
cd cifar

# Basic execution
python main.py --adaptation ours --dataset cifar10

# With custom hyperparameters
python main.py --adaptation ours \
    --dataset cifar10 \
    --n_aug 1 \
    --mt 0.999 \
    --thr 0.4 \
    --ours_alpha 2.0 \
    --lr 0.001 \
    --batch_size 200
```

#### Key Arguments (CIFAR)
| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptation` | `tent` | Adaptation method (`ours` for PAF-KIP) |
| `--dataset` | `cifar10` | Dataset (`cifar10` or `cifar100`) |
| `--n_aug` | `1` | Number of augmentations for EMA model |
| `--mt` | `0.999` | EMA decay rate (teacher momentum) |
| `--thr` | `0.4` | Entropy threshold ratio for PAF filtering |
| `--ours_alpha` | `2.0` | OOD loss weight (negative entropy) |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `200` | Batch size |
| `--data_dir` | `~` | Data directory |

### ImageNet (PAF-KIP)

```bash
cd imagenet

# Basic execution
python main.py --adaptation ours

# With custom hyperparameters
python main.py --adaptation ours \
    --n_aug 1 \
    --mt 0.999 \
    --thr 0.4 \
    --ours_alpha 2.0 \
    --lr 0.001 \
    --batch_size 32
```

#### Key Arguments (ImageNet)
| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptation` | `tent` | Adaptation method (`ours` for PAF-KIP) |
| `--arch` | `Hendrycks2020AugMix` | Model architecture |
| `--n_aug` | `1` | Number of augmentations for EMA model |
| `--mt` | `0.999` | EMA decay rate (teacher momentum) |
| `--thr` | `0.4` | Entropy threshold ratio for PAF filtering |
| `--ours_alpha` | `2.0` | OOD loss weight (negative entropy) |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--data_dir` | `~/imagenet` | Data directory |

## Method Overview

**PAF (Primary-Auxiliary Filtering)**: Entropy-based sample filtering
- ID samples (low entropy): Minimize entropy loss
- OOD samples (high entropy): Maximize entropy loss (negative weight)

**KIP (Knowledge-Integrated Prediction)**: Confidence-based weighted ensemble
- Combines predictions from: model, EMA model, and frozen source model
- Weights based on prediction confidence

## BF16 Support

The implementation uses BF16 (bfloat16) for all operations with numerical stability:
- All model weights converted to BF16
- BatchNorm parameters in BF16
- Safe entropy/logsumexp with epsilon clamping


