"""
PAF-KIP quantization toolkit.

Two layers:

`quantization.ptq`
    Hand-rolled W4A4 fake-quant stack + calibration: AdaRound, BRECQ, QDrop,
    NIPQ (learnable scale + mixed-precision bits), SmoothQuant.

`quantization.mx`
    OCP Microscaling format (MX-FP4 / MX-INT4) — torch.ao-style mapping graph
    with explicit per-layer FakeQuantize sub-modules so `torch.fx.symbolic_trace`
    can lift quantize/dequantize stubs as graph nodes.

Public API (re-exported here for convenience):
    apply_quantization, calibrate_adaround, calibrate_brecq,
    calibrate_nipq, apply_smoothquant, apply_mx_quantization
"""
from .ptq import (
    apply_quantization,
    calibrate_adaround,
    calibrate_brecq,
    calibrate_nipq,
    apply_smoothquant,
    QuantConv2d,
)
from .mx import (
    apply_mx_quantization,
    MXQuantConv2d,
    MXFakeQuantize,
    MXWeightFakeQuant,
    MXActFakeQuantUnfold,
    fx_trace,
    E2M1_LEVELS,
    E2M1_MIDPOINTS,
)

__all__ = [
    # PTQ / hand-rolled
    'apply_quantization', 'calibrate_adaround', 'calibrate_brecq',
    'calibrate_nipq', 'apply_smoothquant', 'QuantConv2d',
    # MX (Microscaling) + torch.fx integration
    'apply_mx_quantization', 'MXQuantConv2d',
    'MXFakeQuantize', 'MXWeightFakeQuant', 'MXActFakeQuantUnfold',
    'fx_trace', 'E2M1_LEVELS', 'E2M1_MIDPOINTS',
]
