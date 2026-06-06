"""
MX (Microscaling, OCP spec) fake-quantization for Conv2d, structured as
torch.ao-style mapping graph.

Each MXQuantConv2d exposes its quantize/dequantize operators as named
sub-modules (`weight_fake_quant`, `act_fake_quant`). When the model is traced
with `torch.fx.symbolic_trace`, the resulting graph contains explicit
`call_module` nodes for the quantize/dequantize stubs, matching the structure
that torch.ao FX quantization produces with QuantStub/DeQuantStub.

Supported element formats:
  - MX-FP4 (E2M1)  : levels {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  - MX-INT4        : signed, levels [-8, 7] (symmetric scale uses ±7)

Per-group shared scale (E8M0 power-of-2 by default) along the convolution
contraction vector C_in * k_h * k_w (im2col view).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Element format primitives
# ---------------------------------------------------------------------------
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
FP4_MAX = 6.0
INT4_QMAX = 7
INT4_QMIN = -8
INT4_MAX_VAL = 7.0  # symmetric-scaling target


def _ste(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Forward = q, backward = identity (straight-through estimator)."""
    return (q - x).detach() + x


def _round_e2m1(x: torch.Tensor, levels: torch.Tensor, mids: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1 representable value (round-half-up at midpoints)."""
    sign = x.sign()
    ax = x.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(ax, mids, right=True)
    return sign * levels[idx]


def _round_int4(x: torch.Tensor) -> torch.Tensor:
    return x.round().clamp(INT4_QMIN, INT4_QMAX)


def _e8m0_scale(max_abs: torch.Tensor, level_max: float) -> torch.Tensor:
    """Smallest power-of-2 such that max_abs / scale <= level_max."""
    in_dtype = max_abs.dtype
    m = max_abs.detach().float()
    ratio = (m / level_max).clamp(min=2.0 ** -126)
    exp = torch.ceil(torch.log2(ratio)).clamp(-127, 127)
    return torch.pow(2.0, exp).to(in_dtype)


# ---------------------------------------------------------------------------
# Fake-quant building blocks (torch.ao-style, exposed as nn.Module)
# ---------------------------------------------------------------------------
class MXFakeQuantize(nn.Module):
    """
    Quantize + dequantize a tensor in MX block format along the last dim.

    Input shape is (..., G) where G == group_size. The shared scale is computed
    over the last dim. The output has the same shape and dtype, holding the
    fake-dequantized values (still floating-point) — exactly what torch.ao's
    quantize→dequantize stub pair produces in the FX graph.
    """

    def __init__(self, fp4: bool = True, use_e8m0: bool = True):
        super().__init__()
        self.fp4 = bool(fp4)
        self.use_e8m0 = bool(use_e8m0)
        # round-to-nearest lookup tables (registered so they move with .to())
        self.register_buffer('_levels', torch.tensor(list(E2M1_LEVELS)))
        self.register_buffer('_midpoints', torch.tensor(list(E2M1_MIDPOINTS)))

    @property
    def level_max(self) -> float:
        return FP4_MAX if self.fp4 else INT4_MAX_VAL

    def forward(self, x_grp: torch.Tensor) -> torch.Tensor:
        max_abs = x_grp.detach().abs().amax(dim=-1, keepdim=True)
        if self.use_e8m0:
            scale = _e8m0_scale(max_abs, self.level_max)
        else:
            scale = (max_abs / self.level_max).clamp(min=1e-8).to(x_grp.dtype)
        x_div = x_grp / scale
        if self.fp4:
            lvls = self._levels.to(device=x_div.device, dtype=x_div.dtype)
            mids = self._midpoints.to(device=x_div.device, dtype=x_div.dtype)
            q = _round_e2m1(x_div, lvls, mids)
        else:
            q = _round_int4(x_div)
        q = _ste(q, x_div)
        return q * scale  # fake-dequantized

    def extra_repr(self):
        fmt = 'fp4-E2M1' if self.fp4 else 'int4'
        sc = 'E8M0' if self.use_e8m0 else 'float'
        return f'{fmt}, scale={sc}'


class MXWeightFakeQuant(nn.Module):
    """MX fake-quant for a conv weight tensor (group along Cin*kh*kw).
    Stateless so torch.fx.symbolic_trace lifts it as a single call_module node."""

    def __init__(self, group_size: int = 32, fp4: bool = True, use_e8m0: bool = True):
        super().__init__()
        self.G = int(group_size)
        self.quant = MXFakeQuantize(fp4=fp4, use_e8m0=use_e8m0)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        Cout, Cin, kh, kw = w.shape
        D = Cin * kh * kw
        w_flat = w.reshape(Cout, D)
        rem = D % self.G
        if rem:
            w_flat = F.pad(w_flat, (0, self.G - rem))
        D_pad = w_flat.size(1)
        w_grp = w_flat.view(Cout, D_pad // self.G, self.G)
        return self.quant(w_grp).view(Cout, D_pad)[:, :D].view(Cout, Cin, kh, kw)

    def extra_repr(self):
        return f'group={self.G}'


class MXActFakeQuantUnfold(nn.Module):
    """MX fake-quant for a conv input. Unfolds to the contraction vector
    (Cin*kh*kw, L), groups along the contraction axis, quantizes per-group,
    and returns the unfolded fake-dequantized tensor of shape (B, Cin*kh*kw, L)."""

    def __init__(self, kernel_size, padding, stride, dilation, group_size: int = 32,
                 fp4: bool = True, use_e8m0: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.G = int(group_size)
        self.quant = MXFakeQuantize(fp4=fp4, use_e8m0=use_e8m0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        x_unf = F.unfold(x, (kh, kw), padding=self.padding,
                         stride=self.stride, dilation=self.dilation)   # [B, D, L]
        B, D, L = x_unf.shape
        rem = D % self.G
        if rem:
            x_unf = F.pad(x_unf, (0, 0, 0, self.G - rem))
        D_pad = x_unf.size(1)
        x_grp = x_unf.view(B, D_pad // self.G, self.G, L).permute(0, 1, 3, 2).contiguous()
        x_dq = self.quant(x_grp)
        x_dq = x_dq.permute(0, 1, 3, 2).contiguous().view(B, D_pad, L)
        return x_dq[:, :D, :]

    def extra_repr(self):
        return f'group={self.G}, kernel={self.kernel_size}'


# ---------------------------------------------------------------------------
# MXQuantConv2d — torch.ao-style mapping pattern
# ---------------------------------------------------------------------------
class MXQuantConv2d(nn.Module):
    """
    Fake-quant Conv2d in MX block-floating-point format. Forward graph:

        x → act_fake_quant (quantize+dequantize) ──┐
        weight → weight_fake_quant ────────────────┴→ matmul → reshape → +bias

    `weight_fake_quant` and `act_fake_quant` are exposed as nn.Module
    sub-attributes so `torch.fx.symbolic_trace` lifts them as `call_module`
    nodes — matching the torch.ao mapping-graph structure.
    """

    def __init__(self, conv: nn.Conv2d, fp4: bool = True, group_size: int = 32,
                 use_e8m0: bool = True, quant_act: bool = True):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.data.clone(),
                                   requires_grad=conv.weight.requires_grad)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone(),
                                     requires_grad=conv.bias.requires_grad)
        else:
            self.register_parameter('bias', None)

        self.weight_fake_quant = MXWeightFakeQuant(group_size=group_size,
                                                   fp4=fp4, use_e8m0=use_e8m0)
        if quant_act:
            self.act_fake_quant = MXActFakeQuantUnfold(
                kernel_size=conv.kernel_size, padding=conv.padding,
                stride=conv.stride, dilation=conv.dilation,
                group_size=group_size, fp4=fp4, use_e8m0=use_e8m0)
        else:
            self.act_fake_quant = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dq = self.weight_fake_quant(self.weight)
        if self.act_fake_quant is None:
            return F.conv2d(x, w_dq, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        x_dq_unf = self.act_fake_quant(x)                       # [B, D, L]
        Cout = self.out_channels
        Cin, kh, kw = self.in_channels, *self.kernel_size
        w_flat = w_dq.view(Cout, Cin * kh * kw)                 # [Cout, D]
        out_unf = torch.matmul(w_flat, x_dq_unf)                # [B, Cout, L]
        H_in, W_in = x.shape[-2], x.shape[-1]
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        out = out_unf.view(out_unf.size(0), Cout, H_out, W_out)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Replace helpers
# ---------------------------------------------------------------------------
def _replace_conv2d_mx(model: nn.Module, fp4: bool, group_size: int,
                       use_e8m0: bool, quant_act: bool, skip_first: bool,
                       _state) -> int:
    n = 0
    for name, mod in model.named_children():
        if isinstance(mod, nn.Conv2d):
            if skip_first and not _state['first_seen']:
                _state['first_seen'] = True
                _state['skipped'].append(name)
                continue
            _state['first_seen'] = True
            q = MXQuantConv2d(mod, fp4=fp4, group_size=group_size,
                              use_e8m0=use_e8m0, quant_act=quant_act)
            q = q.to(mod.weight.device)
            setattr(model, name, q)
            n += 1
        else:
            n += _replace_conv2d_mx(mod, fp4, group_size, use_e8m0, quant_act,
                                    skip_first, _state)
    return n


def apply_mx_quantization(model: nn.Module, fp4: bool = True,
                          group_size: int = 32, use_e8m0: bool = True,
                          quant_act: bool = True, skip_first: bool = True) -> nn.Module:
    """Replace Conv2d layers with MXQuantConv2d (per-layer MX-FP4 or MX-INT4)."""
    state = {'first_seen': False, 'skipped': []}
    n = _replace_conv2d_mx(model, fp4, group_size, use_e8m0, quant_act,
                           skip_first, state)
    fmt = 'MX-FP4(E2M1)' if fp4 else 'MX-INT4'
    e8 = 'E8M0' if use_e8m0 else 'float-scale'
    qa = 'W+A' if quant_act else 'W-only'
    skip = f", skip_first={state['skipped']}" if skip_first else ''
    print(f'[mx] Replaced {n} Conv2d with {fmt}'
          f' [group={group_size}, scale={e8}, {qa}{skip}]')
    return model


# ---------------------------------------------------------------------------
# torch.fx integration: leaf-module tracer so fake-quant ops show up as
# single `call_module` nodes (matches torch.ao QuantStub/FakeQuantize pattern).
# ---------------------------------------------------------------------------
def fx_trace(module: nn.Module):
    """Trace `module` with MX fake-quant modules as leaves.

    The returned GraphModule's graph contains one `call_module` node per
    `weight_fake_quant` / `act_fake_quant` instance — directly analogous to
    torch.ao's quantize/dequantize stubs after `prepare_fx`. Use this to
    serialize / lower to a real MX-INT4 / MX-FP4 inference backend.
    """
    import torch.fx as fx

    class _MXLeafTracer(fx.Tracer):
        def is_leaf_module(self, m, qualname):
            if isinstance(m, (MXFakeQuantize, MXWeightFakeQuant, MXActFakeQuantUnfold)):
                return True
            return super().is_leaf_module(m, qualname)

    tracer = _MXLeafTracer()
    graph = tracer.trace(module)
    return fx.GraphModule(tracer.root, graph)
