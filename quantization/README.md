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
| `--mx_a_bits {4,8}` | MX activation element bits (8 = MXINT8) |

## Results

Mean ACC, 15 corruptions, severity 5. CIFAR: PAF-KIP open-set, gaussian OOD. ImageNet-C: ResNet-50, closed-set.

| config | CIFAR-10 | CIFAR-100 | ImageNet-C |
| --- | --- | --- | --- |
| BF16 | 78.37 | 48.68 | 47.28 |
| global W4A4 | 71.81 | 42.96 | 27.61 |
| global W4A8 | 76.10 | 46.88 | 40.57 |
| MX W4A4 (g16) | 77.73 | 45.62 | 36.08 |
| MX W4A8 (g16) | 78.76 | 47.81 | 45.76 |

MX W4A8 = MXINT4 weights + MXINT8 activations, both OCP MX element types.

## IREE

`torch.export` of MX layers lowers to standard ATen ops and compiles/runs through IREE (`iree-turbine`, llvm-cpu): single layers bit-exact, full MX-quantized WRN-40-2 matches PyTorch logits to 1.4e-6. `fx_trace`'s leaf Q/DQ form is the pattern-matching interface for swapping in real MX kernels (MLIR has `Float4E2M1FN`/`Float8E8M0FNU` upstream). E8M0 exponent floor is −100: `pow(2,−126)` underflows to 0 in expf-based lowerings and produces NaN via 0/0; sub-floor groups quantize to 0 either way.
