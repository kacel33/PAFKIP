# Quantization

## `quantization.ptq`

W4A4 fake-quant `QuantConv2d` and calibration:
- AdaRound (Nagel '20)
- BRECQ (Li '21)
- QDrop (Wei '22)
- NIPQ (Shin '23)
- SmoothQuant (Xiao '23)

Activation granularity: `per_tensor`, `per_channel`, `per_sample`. Dynamic percentile clipping, STE weight rounding. Entry: `apply_quantization(...)`.

## `quantization.mx`

MX (OCP Microscaling) fake-quant for Conv2d.

- MX-FP4: E2M1 levels `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`
- MX-INT4: signed 4-bit
- Group along `C_in * k_h * k_w` (im2col contraction)
- E8M0 power-of-2 per-group scale (default) or float
- Round-to-nearest, round-half-up at midpoints

`MXQuantConv2d.forward`:
```
weight_fake_quant(weight) ──┐
act_fake_quant(x) ──────────┴→ matmul → reshape → +bias
```

`weight_fake_quant` and `act_fake_quant` are `nn.Module` submodules. `fx_trace` (custom leaf tracer) lifts them as `call_module` nodes — same shape as `torch.ao.quantization.prepare_fx` output.

```python
from quantization import apply_mx_quantization, fx_trace

apply_mx_quantization(model, fp4=True, group_size=32)
gm = fx_trace(model.some_conv_layer)
for node in gm.graph.nodes:
    print(node.op, node.target)
# call_module    weight_fake_quant
# call_module    act_fake_quant
# call_function  matmul
```

## API

```python
from quantization import (
    apply_quantization,        # W4A4 PTQ entry
    calibrate_adaround, calibrate_brecq, calibrate_nipq,
    apply_smoothquant,
    apply_mx_quantization,     # MX-FP4 / MX-INT4
    fx_trace,                  # torch.fx leaf tracer
)
```

CLI flags (`cifar/main.py`):

| flag | meaning |
| --- | --- |
| `--quantize --w_bits 4 --a_bits 4` | hand-rolled W4A4 |
| `--mse_weight --act_percentile 0.999 --skip_first_conv` | scale knobs |
| `--adaround` / `--brecq` / `--nipq` | calibration |
| `--qdrop_p 0.5` | QDrop |
| `--act_granularity {per_tensor,per_channel,per_sample}` | act grouping |
| `--nipq_target_bit 4.5` | mixed-precision bit budget |
| `--smoothquant --sq_alpha 0.5` | BN→conv migration |
| `--mx_quantize --mx_format {fp4,int4} --mx_group_size {16,32}` | MX block |
| `--mx_no_e8m0` | float per-group scale |
| `--mx_no_skip_first` | quantize first Conv2d too |

## Results

CIFAR-10-C, PAF-KIP, gaussian OOD, BF16 baseline 78.37 %.

| config | mean ACC | Δ vs BF16 | AUROC |
| --- | --- | --- | --- |
| MX-INT4 g=16 | 77.73 | −0.64 | 99.75 |
| MX-INT4 g=32 | 77.50 | −0.87 | 99.75 |
| MX-FP4 g=16 | 76.77 | −1.60 | 99.73 |
| MX-FP4 g=32 | 76.80 | −1.57 | 99.76 |
| MX-INT4 g=32 strict | 76.72 | −1.65 | 99.75 |
| MX-FP4 g=32 strict | 75.33 | −3.04 | 99.69 |
