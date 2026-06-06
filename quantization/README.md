# PAF-KIP Quantization Toolkit

Two complementary modules used by the PAF-KIP test-time adaptation experiments.

## `quantization.ptq` — hand-rolled W4A4 PTQ stack
Fake-quantization Conv2d wrapper (`QuantConv2d`) plus calibration routines:
- **AdaRound** (Nagel et al., ICML 2020) — learnable per-weight rounding.
- **BRECQ** (Li et al., ICLR 2021) — block-wise reconstruction.
- **QDrop** (Wei et al., 2022) — stochastic quantization drop during calibration.
- **NIPQ** (Shin et al., CVPR 2023) — learnable activation scale + mixed-precision bit allocation.
- **SmoothQuant** (Xiao et al., ICML 2023) — channel-wise activation→weight migration via the preceding BatchNorm.

Activations support per-tensor / per-channel / per-sample granularity, dynamic percentile clipping, and STE-based weight rounding. See `apply_quantization(...)` for the entry point.

## `quantization.mx` — OCP Microscaling (MX-FP4 / MX-INT4)
Hardware-faithful MX block-floating-point fake-quant for Conv2d. Element formats:
- **MX-FP4** — E2M1 elements `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
- **MX-INT4** — signed 4-bit integer.

Shared per-group scale (E8M0 power-of-2 by default) along the convolution contraction vector (`C_in * k_h * k_w` after `F.unfold`). Round-to-nearest with round-half-up tie semantics for E2M1.

### torch.ao-style mapping graph
Each `MXQuantConv2d` exposes its quantize/dequantize operators as named submodules:

```
forward(x) =  weight_fake_quant(weight) ──┐
              act_fake_quant(x) ──────────┴→ matmul → reshape → +bias
```

Tracing the model with the provided leaf-aware tracer produces an FX graph with explicit `call_module` nodes for the fake-quant stubs — the same shape that `torch.ao.quantization.prepare_fx` yields with QuantStub / DeQuantStub:

```python
from quantization import apply_mx_quantization, fx_trace

model = apply_mx_quantization(model, fp4=True, group_size=32)   # MX-FP4 g=32
gm = fx_trace(model.some_conv_layer)
for node in gm.graph.nodes:
    print(node.op, node.target, '->', node.name)
# call_module weight_fake_quant -> weight_fake_quant
# call_module act_fake_quant    -> act_fake_quant
# call_function matmul          -> matmul
# ...
```

This makes the model lowerable to a real MX-INT4 / MX-FP4 inference backend (Blackwell, MX-capable NPUs) by replacing the leaf fake-quant modules with hardware kernels.

## Quick reference

```python
from quantization import (
    apply_quantization,        # PTQ entry — W4A4 + AdaRound / BRECQ / NIPQ / SmoothQuant
    calibrate_adaround, calibrate_brecq, calibrate_nipq,
    apply_smoothquant,
    apply_mx_quantization,     # MX-FP4 or MX-INT4
    fx_trace,                  # torch.fx graph with leaf fake-quant nodes
)
```

CLI flags (see `cifar/main.py`):

| flag | meaning |
| --- | --- |
| `--quantize --w_bits 4 --a_bits 4` | enable hand-rolled W4A4 |
| `--mse_weight --act_percentile 0.999 --skip_first_conv` | activation/weight calibration knobs |
| `--adaround` / `--brecq` / `--nipq` | enable a calibration routine |
| `--smoothquant --sq_alpha 0.5` | SmoothQuant BN→conv migration |
| `--mx_quantize --mx_format {fp4,int4} --mx_group_size {16,32}` | MX block format |
| `--mx_no_e8m0` | use float per-group scale instead of E8M0 (debug) |
| `--mx_no_skip_first` | quantize the first Conv2d too (strict HW-faithful) |

## Empirical summary (PAF-KIP / CIFAR-10-C / gaussian OOD)

`BF16` baseline = **78.37 %** mean accuracy.

| config | mean ACC | Δ vs BF16 | AUROC | deployable |
| --- | --- | --- | --- | --- |
| MX-INT4 g=16 (skip first) | 77.73 | −0.64 | 99.75 | ✅ |
| MX-INT4 g=32 (skip first) | 77.50 | −0.87 | 99.75 | ✅ |
| MX-FP4 g=16 (skip first)  | 76.77 | −1.60 | 99.73 | ✅ |
| MX-FP4 g=32 (skip first)  | 76.80 | −1.57 | 99.76 | ✅ |
| MX-INT4 g=32 strict (no skip) | 76.72 | −1.65 | 99.75 | ✅ |
| MX-FP4 g=32 strict (no skip)  | 75.33 | −3.04 | 99.69 | ⚠️ |

MX-INT4 outperforms MX-FP4 by ~1 pp on this CNN/TTA setup because the E2M1 grid is sparse in the 4–6 range, while CNN BN-normalized activations populate that range densely. MX-FP4 is the better fit for outlier-heavy LLM activations.
