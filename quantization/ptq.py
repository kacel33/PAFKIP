import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_ste(x):
    return (x.round() - x).detach() + x


def _mse_optimal_per_channel_scale(w: torch.Tensor, qmin: int, qmax: int,
                                   ratios=None) -> torch.Tensor:
    """Per-channel symmetric scale by MSE grid search (BRECQ pre-step)."""
    if ratios is None:
        ratios = torch.linspace(0.5, 1.0, 26)
    w_flat = w.reshape(w.size(0), -1).float()
    max_abs = w_flat.abs().amax(dim=1, keepdim=True)
    best_scale = None
    best_err = None
    for r in ratios:
        r = float(r)
        scale = (max_abs * r / qmax).clamp(min=1e-8)
        w_q = (w_flat / scale).round().clamp(qmin, qmax) * scale
        err = (w_q - w_flat).pow(2).mean(dim=1)
        if best_err is None:
            best_err = err
            best_scale = scale.squeeze(-1).clone()
        else:
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_scale = torch.where(mask, scale.squeeze(-1), best_scale)
    return best_scale


# AdaRound rectified-sigmoid params (Nagel et al., ICML 2020)
_ADA_ZETA = 1.1
_ADA_GAMMA = -0.1


def _rectified_sigmoid(alpha):
    return (torch.sigmoid(alpha) * (_ADA_ZETA - _ADA_GAMMA) + _ADA_GAMMA).clamp(0.0, 1.0)


class QuantConv2d(nn.Module):
    """
    Fake-quantized Conv2d with options:
      act_percentile<1: clip activations at percentile (outlier-robust).
      mse_weight: per-channel weight scale via MSE grid search.
      adaround: enable learnable per-weight rounding (call init_adaround() then optimise alpha).
    Set self._fp_mode=True to bypass quantisation (for calibration capture).
    """

    def __init__(self, conv: nn.Conv2d, w_bits: int = 4, a_bits: int = 4,
                 act_percentile: float = 1.0, mse_weight: bool = False,
                 adaround: bool = False, act_granularity: str = 'per_tensor',
                 nipq: bool = False):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.data.clone(), requires_grad=conv.weight.requires_grad)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone(), requires_grad=conv.bias.requires_grad)
        else:
            self.register_parameter('bias', None)

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_qmin = -(2 ** (w_bits - 1))
        self.w_qmax = 2 ** (w_bits - 1) - 1
        self.a_qmin = -(2 ** (a_bits - 1))
        self.a_qmax = 2 ** (a_bits - 1) - 1

        self.act_percentile = float(act_percentile)
        self.act_granularity = act_granularity  # 'per_tensor' | 'per_channel' | 'per_sample'
        assert act_granularity in ('per_tensor', 'per_channel', 'per_sample')
        self.adaround_enabled = bool(adaround)
        self.adaround_soft = True
        self._fp_mode = False
        self.qdrop_p = 0.0  # set >0 during calibration to enable QDrop
        self._qdrop_active = False  # gate set only during calibration
        self.alpha = None  # nn.Parameter, created by init_adaround()

        # NIPQ (learnable per-tensor activation scale + continuous bit-width)
        self.nipq_enabled = bool(nipq)
        self._nipq_noise = False       # True during calib (pseudo-quant noise), else hard quant
        self.nipq_bmin = 2.0
        self.nipq_bmax = 8.0
        self.a_alpha = None            # nn.Parameter, learnable activation clip range
        self.a_bit_param = None        # nn.Parameter, maps to continuous bit via sigmoid
        self.nipq_dynamic_infer = False  # at inference, use dynamic per-tensor scale (keep learned bits)
        self.nipq_dyn_percentile = 0.999
        self.register_buffer('a_bit_int', torch.tensor(float(a_bits)))  # snapped bit for inference
        self.register_buffer('act_numel', torch.tensor(0.0))            # MAC-cost weight (set at init)

        w = self.weight.data
        if mse_weight:
            w_scale = _mse_optimal_per_channel_scale(w, self.w_qmin, self.w_qmax)
        else:
            w_flat = w.reshape(w.size(0), -1)
            max_abs = w_flat.abs().amax(dim=1)
            w_scale = (max_abs / self.w_qmax).clamp(min=1e-8)
        self.register_buffer('w_scale', w_scale)

    def init_adaround(self):
        """Initialise the learnable rounding parameter alpha so that h(alpha) ≈ fractional part."""
        with torch.no_grad():
            scale = self.w_scale.view(-1, 1, 1, 1).to(self.weight.dtype)
            w_div_s = self.weight.data / scale
            frac = w_div_s - w_div_s.floor()  # in [0, 1)
            p = ((frac - _ADA_GAMMA) / (_ADA_ZETA - _ADA_GAMMA)).clamp(min=1e-6, max=1 - 1e-6)
            alpha_init = torch.log(p / (1 - p))
        self.alpha = nn.Parameter(alpha_init.detach().clone(), requires_grad=True)

    def freeze_adaround(self):
        """Lock in the hard rounding decision from the learned alpha."""
        if self.alpha is not None:
            self.adaround_soft = False
            self.alpha.requires_grad_(False)

    # ---- NIPQ (Shin et al., CVPR'23): learnable act scale + continuous bit ----
    def init_nipq(self, init_alpha: float, init_bit: float = 4.0):
        frac = (init_bit - self.nipq_bmin) / (self.nipq_bmax - self.nipq_bmin)
        frac = min(max(frac, 1e-3), 1 - 1e-3)
        p = math.log(frac / (1 - frac))
        dev = self.weight.device
        # shape [1] (not 0-dim) so EMA update's `param[:]` indexing in ours.py works
        self.a_alpha = nn.Parameter(torch.tensor([float(max(init_alpha, 1e-4))], device=dev))
        self.a_bit_param = nn.Parameter(torch.tensor([float(p)], device=dev))

    def current_a_bit(self):
        return self.nipq_bmin + (self.nipq_bmax - self.nipq_bmin) * torch.sigmoid(self.a_bit_param)

    def snap_nipq(self, allowed=(4.0, 8.0)):
        """Round learned continuous bit to nearest allowed bit; freeze params for inference."""
        with torch.no_grad():
            b = float(self.current_a_bit())
            snapped = min(allowed, key=lambda c: abs(c - b))
            self.a_bit_int = torch.tensor(float(snapped), device=self.a_bit_int.device)
        self._nipq_noise = False
        if self.a_alpha is not None:
            self.a_alpha.requires_grad_(False)
            self.a_bit_param.requires_grad_(False)
        return float(self.a_bit_int), b

    def _quantize_act_nipq(self, x):
        # a_alpha/a_bit_param are shape [1]; reshape to broadcast over [B,C,H,W]
        alpha = self.a_alpha.abs().clamp(min=1e-5).to(x.dtype).view(1, 1, 1, 1)
        if self._nipq_noise:
            b = self.current_a_bit().to(x.dtype).view(1, 1, 1, 1)
            qmax = (2.0 ** (b - 1)) - 1
            delta = alpha / qmax
            x_clip = torch.minimum(torch.maximum(x, -alpha), alpha)
            noise = (torch.rand_like(x) - 0.5) * delta  # pseudo-quant noise
            return x_clip + noise
        else:
            b = self.a_bit_int.to(x.dtype)
            qmax = (2.0 ** (b - 1)) - 1
            qmin = -(2.0 ** (b - 1))
            if self.nipq_dynamic_infer:
                # keep learned bit, but recompute scale dynamically (percentile) per batch
                xa = x.detach().abs().reshape(-1).float()
                npx = xa.numel()
                if npx > 1_000_000:
                    xa = xa[::(npx // 1_000_000 + 1)]; npx = xa.numel()
                k = max(1, min(int(npx * self.nipq_dyn_percentile), npx))
                alpha = xa.kthvalue(k).values.to(x.dtype).clamp(min=1e-5).view(1, 1, 1, 1)
            delta = alpha / qmax
            x_clip = torch.minimum(torch.maximum(x, -alpha), alpha)
            x_q = _round_ste(x_clip / delta).clamp(qmin, qmax)
            return x_q * delta

    def _quantize_weight(self):
        scale = self.w_scale.view(-1, 1, 1, 1).to(self.weight.dtype)
        if self.adaround_enabled and self.alpha is not None:
            w_div_s = self.weight / scale
            w_floor = w_div_s.floor()
            if self.adaround_soft:
                h = _rectified_sigmoid(self.alpha.to(self.weight.dtype))
            else:
                h = (self.alpha > 0).to(self.weight.dtype)
            w_q = (w_floor + h).clamp(self.w_qmin, self.w_qmax)
        else:
            w_q = _round_ste(self.weight / scale).clamp(self.w_qmin, self.w_qmax)
        return w_q * scale

    def _act_clip_value(self, x):
        """Compute clip threshold per requested granularity. Returns tensor that broadcasts to x's shape."""
        if self.act_granularity == 'per_tensor':
            if self.act_percentile >= 1.0:
                return x.detach().abs().amax()
            x_abs = x.detach().abs().reshape(-1).float()
            n = x_abs.numel()
            if n > 1_000_000:
                stride = n // 1_000_000 + 1
                x_abs = x_abs[::stride]
                n = x_abs.numel()
            k = max(1, min(int(n * self.act_percentile), n))
            return x_abs.kthvalue(k).values
        elif self.act_granularity == 'per_channel':
            # one scale per channel C of [B, C, H, W]
            x_abs = x.detach().abs().float()
            B, C = x_abs.size(0), x_abs.size(1)
            x_flat = x_abs.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
            n = x_flat.size(1)
            if self.act_percentile >= 1.0:
                clip = x_flat.amax(dim=1)
            else:
                if n > 100_000:
                    stride = n // 100_000 + 1
                    x_flat = x_flat[:, ::stride]
                    n = x_flat.size(1)
                k = max(1, min(int(n * self.act_percentile), n))
                clip = x_flat.kthvalue(k, dim=1).values
            return clip.view(1, -1, 1, 1)
        else:  # per_sample
            x_abs = x.detach().abs().float()
            B = x_abs.size(0)
            x_flat = x_abs.reshape(B, -1)  # [B, C*H*W]
            n = x_flat.size(1)
            if self.act_percentile >= 1.0:
                clip = x_flat.amax(dim=1)
            else:
                if n > 100_000:
                    stride = n // 100_000 + 1
                    x_flat = x_flat[:, ::stride]
                    n = x_flat.size(1)
                k = max(1, min(int(n * self.act_percentile), n))
                clip = x_flat.kthvalue(k, dim=1).values
            return clip.view(-1, 1, 1, 1)

    def _quantize_act(self, x):
        clip = self._act_clip_value(x).to(x.dtype).clamp(min=1e-8)
        scale = clip / self.a_qmax
        if self.act_granularity == 'per_tensor':
            x_clipped = x.clamp(-clip, clip)
        else:
            x_clipped = torch.minimum(torch.maximum(x, -clip), clip)
        x_q = _round_ste(x_clipped / scale).clamp(self.a_qmin, self.a_qmax)
        return x_q * scale

    def forward(self, x):
        if self._fp_mode:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.nipq_enabled and self.a_alpha is not None:
            x_q = self._quantize_act_nipq(x)
        else:
            x_q = self._quantize_act(x)
        w_q = self._quantize_weight()
        # QDrop (Wei '22): per-element stochastic drop of activation quant during calibration
        if self.qdrop_p > 0 and self._qdrop_active:
            mask = (torch.rand_like(x) > self.qdrop_p).to(x.dtype)
            x_q = mask * x_q + (1 - mask) * x
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, w_bits={self.w_bits}, a_bits={self.a_bits}, "
                f"act_pct={self.act_percentile}, adaround={self.adaround_enabled}")


@torch.no_grad()
def _smooth_bn_conv(bn: nn.BatchNorm2d, convs, act_max: torch.Tensor,
                    alpha: float, s_clamp=(1e-3, 1e3)):
    """
    Migrate per-input-channel activation range into the preceding BN.
    s_c = act_max_c^alpha / w_max_c^(1-alpha);  BN.{w,b} /= s;  conv.w[:, c] *= s.
    `convs` share the same input channels (e.g. conv1 and convShortcut after bn1).
    """
    eps = 1e-5
    w_max = None
    for conv in convs:
        wm = conv.weight.detach().abs().amax(dim=(0, 2, 3))  # [C_in]
        w_max = wm if w_max is None else torch.maximum(w_max, wm)
    act_max = act_max.to(w_max.dtype)
    s = (act_max.clamp(min=eps) ** alpha) / (w_max.clamp(min=eps) ** (1 - alpha))
    s = s.clamp(min=s_clamp[0], max=s_clamp[1])

    bn.weight.data.div_(s)
    bn.bias.data.div_(s)
    for conv in convs:
        conv.weight.data.mul_(s.view(1, -1, 1, 1))
    return s


def apply_smoothquant(model: nn.Module, calib_inputs: torch.Tensor,
                      alpha: float = 0.5, block_class_name: str = 'BasicBlock',
                      chunk_size: int = 128, verbose: bool = True) -> nn.Module:
    """
    SmoothQuant (Xiao '23) for CNNs. Must be called on the FP model BEFORE
    apply_quantization. For each WRN BasicBlock, migrates activation outliers
    from conv inputs into the preceding BatchNorm (bn1 feeds conv1 [+ convShortcut],
    bn2 feeds conv2). Mathematically output-preserving at application time.
    """
    blocks = [(n, m) for n, m in model.named_modules()
              if m.__class__.__name__ == block_class_name]
    if not blocks:
        print(f"[smoothquant] no {block_class_name} found"); return model

    # --- collect per-input-channel activation max for conv1/conv2 inputs ---
    act_max = {}
    hooks = []

    def make_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach()
            m = x.abs().amax(dim=(0, 2, 3)).float()  # [C_in]
            act_max[key] = m if key not in act_max else torch.maximum(act_max[key], m)
        return hook

    for name, block in blocks:
        hooks.append(block.conv1.register_forward_hook(make_hook(name + '|conv1')))
        hooks.append(block.conv2.register_forward_hook(make_hook(name + '|conv2')))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, calib_inputs.size(0), chunk_size):
            _ = model(calib_inputs[i:i + chunk_size])
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    # --- absorb smoothing ---
    n_pairs = 0
    for name, block in blocks:
        _smooth_bn_conv(block.bn2, [block.conv2], act_max[name + '|conv2'], alpha)
        n_pairs += 1
        convs1 = [block.conv1]
        if getattr(block, 'convShortcut', None) is not None:
            convs1.append(block.convShortcut)
        _smooth_bn_conv(block.bn1, convs1, act_max[name + '|conv1'], alpha)
        n_pairs += 1

    if verbose:
        print(f"[smoothquant] migrated {n_pairs} BN->conv pairs over "
              f"{len(blocks)} {block_class_name}s (alpha={alpha})")
    return model


def _replace_conv2d(model: nn.Module, w_bits: int, a_bits: int,
                    act_percentile: float, mse_weight: bool, adaround: bool,
                    skip_first: bool, act_granularity: str, nipq: bool, _state) -> int:
    n = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            if skip_first and not _state['first_seen']:
                _state['first_seen'] = True
                _state['skipped'].append(name)
                continue
            _state['first_seen'] = True
            q = QuantConv2d(module, w_bits=w_bits, a_bits=a_bits,
                            act_percentile=act_percentile, mse_weight=mse_weight,
                            adaround=adaround, act_granularity=act_granularity,
                            nipq=nipq)
            q = q.to(module.weight.device)
            setattr(model, name, q)
            n += 1
        else:
            n += _replace_conv2d(module, w_bits, a_bits, act_percentile,
                                 mse_weight, adaround, skip_first,
                                 act_granularity, nipq, _state)
    return n


def apply_quantization(model: nn.Module, w_bits: int = 4, a_bits: int = 4,
                       act_percentile: float = 1.0, mse_weight: bool = False,
                       skip_first: bool = False, adaround: bool = False,
                       act_granularity: str = 'per_tensor', nipq: bool = False) -> nn.Module:
    state = {'first_seen': False, 'skipped': []}
    n = _replace_conv2d(model, w_bits, a_bits, act_percentile, mse_weight,
                        adaround, skip_first, act_granularity, nipq, state)
    extras = []
    if act_percentile < 1.0: extras.append(f"act_pct={act_percentile}")
    if act_granularity != 'per_tensor': extras.append(f"act={act_granularity}")
    if mse_weight: extras.append("mse_weight")
    if adaround: extras.append("adaround")
    if nipq: extras.append("nipq")
    if skip_first: extras.append(f"skip_first={state['skipped']}")
    extra_str = (" [" + ", ".join(extras) + "]") if extras else ""
    print(f"[quantization] Replaced {n} Conv2d with QuantConv2d (W{w_bits}A{a_bits}){extra_str}")
    return model


def _set_qdrop(model: nn.Module, p: float, active: bool):
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.qdrop_p = p
            m._qdrop_active = active


def calibrate_adaround(model: nn.Module, calib_inputs: torch.Tensor,
                       n_iters: int = 2000, lr: float = 4e-3,
                       lambda_reg: float = 0.01, warmup: float = 0.2,
                       beta_start: float = 20.0, beta_end: float = 2.0,
                       qdrop_p: float = 0.0, verbose: bool = True):
    """
    Per-layer AdaRound calibration (Nagel et al., ICML 2020) via single-layer
    reconstruction loss. Captures FP input/output for each AdaRound layer once,
    then optimises alpha per layer to minimise ||q(W)·x - W·x||² + λ·R(alpha).
    """
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, QuantConv2d) and m.adaround_enabled]
    if not layers:
        print("[adaround] No AdaRound layers found"); return
    if verbose:
        print(f"[adaround] {len(layers)} layers, {calib_inputs.size(0)} samples, "
              f"{n_iters} iters/layer, lr={lr}, λ={lambda_reg}")

    for _, m in layers:
        m.init_adaround()

    # --- 1) capture FP input/output via hooks ---
    captured = {}
    hooks = []
    for name, layer in layers:
        def make_hook(n):
            def hook(mod, inp, out):
                captured[n] = (inp[0].detach(), out.detach())
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name)))

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = True
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(calib_inputs)
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = False
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    # --- 2) per-layer optimisation ---
    if qdrop_p > 0:
        _set_qdrop(model, qdrop_p, True)
    warmup_iters = int(n_iters * warmup)
    for idx, (name, layer) in enumerate(layers):
        x_fp, y_fp = captured[name]
        layer.adaround_soft = True
        opt = torch.optim.Adam([layer.alpha], lr=lr)

        init_loss = None
        for it in range(n_iters):
            if it < warmup_iters:
                reg_weight = 0.0
                beta = beta_start
            else:
                t = (it - warmup_iters) / max(1, n_iters - warmup_iters)
                beta = beta_start + t * (beta_end - beta_start)
                reg_weight = lambda_reg

            x_q = layer._quantize_act(x_fp)
            w_q = layer._quantize_weight()
            y_q = F.conv2d(x_q, w_q, layer.bias, layer.stride, layer.padding,
                           layer.dilation, layer.groups)
            recon_loss = (y_q - y_fp).pow(2).mean()

            if reg_weight > 0:
                h = _rectified_sigmoid(layer.alpha)
                reg = (1 - (2 * h - 1).abs().pow(beta)).sum()
                loss = recon_loss + reg_weight * reg
            else:
                loss = recon_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it == 0:
                init_loss = recon_loss.item()
            if verbose and (it + 1) % max(1, n_iters // 4) == 0:
                print(f"  [{idx+1:02d}/{len(layers)} {name}] iter={it+1}/{n_iters} "
                      f"recon={recon_loss.item():.4e} (init {init_loss:.4e})")

        layer.freeze_adaround()
        # release captured tensors for this layer
        del captured[name]

    if qdrop_p > 0:
        _set_qdrop(model, 0.0, False)
    captured.clear()
    if verbose:
        print("[adaround] done")


def _find_brecq_blocks(model: nn.Module, block_class_name: str = 'BasicBlock'):
    """Find target blocks (e.g. BasicBlock instances) that contain AdaRound conv layers."""
    blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == block_class_name:
            convs = [m for m in module.modules()
                     if isinstance(m, QuantConv2d) and m.adaround_enabled]
            if convs:
                blocks.append((name, module, convs))
    return blocks


def calibrate_brecq(model: nn.Module, calib_inputs: torch.Tensor,
                    n_iters: int = 10000, lr: float = 4e-3,
                    lambda_reg: float = 0.01, warmup: float = 0.2,
                    beta_start: float = 20.0, beta_end: float = 2.0,
                    block_class_name: str = 'BasicBlock',
                    qdrop_p: float = 0.0, verbose: bool = True):
    """
    Block-wise reconstruction (BRECQ, Li et al., ICLR 2021).
    All AdaRound alphas inside one BasicBlock are jointly optimised against the
    block's FP output. Falls back to per-layer reconstruction for AdaRound layers
    that are not contained in any BasicBlock.
    """
    blocks = _find_brecq_blocks(model, block_class_name)

    in_blocks = set()
    for _, _, convs in blocks:
        for c in convs:
            in_blocks.add(id(c))
    standalone = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, QuantConv2d) and m.adaround_enabled
                  and id(m) not in in_blocks]

    if not blocks and not standalone:
        print("[brecq] No AdaRound layers found"); return

    if verbose:
        n_in_blocks = sum(len(c) for _, _, c in blocks)
        print(f"[brecq] {len(blocks)} {block_class_name}s ({n_in_blocks} convs joint), "
              f"{len(standalone)} standalone convs, {calib_inputs.size(0)} samples, "
              f"{n_iters} iters/block, lr={lr}, λ={lambda_reg}")

    for _, _, convs in blocks:
        for c in convs:
            c.init_adaround()
    for _, c in standalone:
        c.init_adaround()

    # --- capture FP block I/O and standalone layer I/O ---
    captured = {}
    hooks = []
    for name, block, _ in blocks:
        def make_hook(n):
            def hook(mod, inp, out):
                captured[n] = (inp[0].detach(), out.detach())
            return hook
        hooks.append(block.register_forward_hook(make_hook(name)))
    for name, layer in standalone:
        def make_hook(n):
            def hook(mod, inp, out):
                captured[n] = (inp[0].detach(), out.detach())
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name)))

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = True
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(calib_inputs)
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = False
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    model.eval()  # keep BN in eval mode during BRECQ

    warmup_iters = int(n_iters * warmup)
    if qdrop_p > 0:
        _set_qdrop(model, qdrop_p, True)

    # --- per-block joint optimisation ---
    for idx, (name, block, convs) in enumerate(blocks):
        x_fp, y_fp = captured[name]
        for c in convs:
            c.adaround_soft = True

        opt = torch.optim.Adam([c.alpha for c in convs], lr=lr)

        init_loss = None
        for it in range(n_iters):
            if it < warmup_iters:
                reg_weight, beta = 0.0, beta_start
            else:
                t = (it - warmup_iters) / max(1, n_iters - warmup_iters)
                beta = beta_start + t * (beta_end - beta_start)
                reg_weight = lambda_reg

            y_q = block(x_fp)
            recon_loss = (y_q - y_fp).pow(2).mean()

            if reg_weight > 0:
                reg = 0
                for c in convs:
                    h = _rectified_sigmoid(c.alpha)
                    reg = reg + (1 - (2 * h - 1).abs().pow(beta)).sum()
                loss = recon_loss + reg_weight * reg
            else:
                loss = recon_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it == 0:
                init_loss = recon_loss.item()
            if verbose and (it + 1) % max(1, n_iters // 4) == 0:
                print(f"  [{idx+1:02d}/{len(blocks)} {name}] iter={it+1}/{n_iters} "
                      f"recon={recon_loss.item():.4e} (init {init_loss:.4e})")

        for c in convs:
            c.freeze_adaround()
        del captured[name]

    # --- per-layer fallback for standalone convs ---
    for idx, (name, layer) in enumerate(standalone):
        x_fp, y_fp = captured[name]
        layer.adaround_soft = True
        opt = torch.optim.Adam([layer.alpha], lr=lr)
        init_loss = None
        for it in range(n_iters):
            if it < warmup_iters:
                reg_weight, beta = 0.0, beta_start
            else:
                t = (it - warmup_iters) / max(1, n_iters - warmup_iters)
                beta = beta_start + t * (beta_end - beta_start)
                reg_weight = lambda_reg
            x_q = layer._quantize_act(x_fp)
            w_q = layer._quantize_weight()
            y_q = F.conv2d(x_q, w_q, layer.bias, layer.stride, layer.padding,
                           layer.dilation, layer.groups)
            recon_loss = (y_q - y_fp).pow(2).mean()
            if reg_weight > 0:
                h = _rectified_sigmoid(layer.alpha)
                reg = (1 - (2 * h - 1).abs().pow(beta)).sum()
                loss = recon_loss + reg_weight * reg
            else:
                loss = recon_loss
            opt.zero_grad(); loss.backward(); opt.step()
            if it == 0:
                init_loss = recon_loss.item()
            if verbose and (it + 1) % max(1, n_iters // 4) == 0:
                print(f"  [standalone {idx+1}/{len(standalone)} {name}] iter={it+1}/{n_iters} "
                      f"recon={recon_loss.item():.4e} (init {init_loss:.4e})")
        layer.freeze_adaround()
        del captured[name]

    if qdrop_p > 0:
        _set_qdrop(model, 0.0, False)
    captured.clear()
    if verbose:
        print("[brecq] done")


def calibrate_nipq(model: nn.Module, calib_inputs: torch.Tensor,
                   n_iters: int = 2000, lr: float = 1e-2,
                   target_bit: float = 4.5, lambda_bit: float = 0.05,
                   chunk_size: int = 128, allowed_bits=(4.0, 8.0),
                   dynamic_infer: bool = False, verbose: bool = True):
    """
    NIPQ-style calibration (Shin et al., CVPR'23) for activation quantization.
    Learns, per QuantConv2d, a per-tensor activation clip range (a_alpha) and a
    continuous activation bit-width (a_bit_param) via pseudo-quantization noise,
    using output distillation to the FP model plus a MAC-weighted bit budget.
    Weights stay at fixed per-channel INT4. After training, bits are snapped to
    `allowed_bits` (deployable mixed precision: per-layer A4/A8).
    """
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, QuantConv2d) and m.nipq_enabled]
    if not layers:
        print("[nipq] no NIPQ layers found"); return

    # --- capture per-layer FP activation absmax (init alpha) and act numel (cost) ---
    stats = {}
    hooks = []

    def make_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach()
            amax = x.abs().amax().float()
            if key not in stats:
                stats[key] = [amax, float(x.numel()) / x.size(0)]  # [absmax, per-sample numel]
            else:
                stats[key][0] = torch.maximum(stats[key][0], amax)
        return hook

    name_of = {id(m): n for n, m in layers}
    for n, m in layers:
        hooks.append(m.register_forward_hook(make_hook(n)))

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = True
    was_training = model.training
    model.eval()
    fp_logits = []
    with torch.no_grad():
        for i in range(0, calib_inputs.size(0), chunk_size):
            fp_logits.append(model(calib_inputs[i:i + chunk_size]).detach())
    fp_logits = torch.cat(fp_logits, 0)
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m._fp_mode = False
    for h in hooks:
        h.remove()

    # init learnable params + cost weights
    total_numel = sum(stats[n][1] for n, _ in layers)
    for n, m in layers:
        m.init_nipq(init_alpha=float(stats[n][0]), init_bit=target_bit)
        m.act_numel = torch.tensor(stats[n][1] / total_numel, device=m.a_alpha.device)
        m._nipq_noise = True

    params = []
    for _, m in layers:
        params += [m.a_alpha, m.a_bit_param]
    opt = torch.optim.Adam(params, lr=lr)

    if verbose:
        print(f"[nipq] {len(layers)} layers, {calib_inputs.size(0)} samples, "
              f"{n_iters} iters, target_bit={target_bit}, lambda={lambda_bit}, "
              f"allowed={allowed_bits}")

    model.eval()  # BN in eval; only a_alpha/a_bit_param require grad
    N = calib_inputs.size(0)
    for it in range(n_iters):
        idx = torch.randperm(N, device=calib_inputs.device)[:chunk_size]
        xb = calib_inputs[idx]
        q_logits = model(xb)
        recon = (q_logits - fp_logits[idx]).pow(2).mean()
        bit_cost = sum(m.act_numel * m.current_a_bit() for _, m in layers)
        budget = torch.relu(bit_cost - target_bit)
        loss = recon + lambda_bit * budget
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (it + 1) % max(1, n_iters // 5) == 0:
            avg_bit = float(sum(m.act_numel * m.current_a_bit() for _, m in layers))
            print(f"  [nipq] iter={it+1}/{n_iters} recon={recon.item():.4e} "
                  f"avg_bit={avg_bit:.3f}")

    # --- global budget-constrained allocation to {lo, hi} bits ---
    # Start everyone at lo bit; greedily promote highest-learned-bit layers to hi
    # while MAC-weighted effective bit stays within target_bit.
    lo, hi = min(allowed_bits), max(allowed_bits)
    learned = {n: float(m.current_a_bit()) for n, m in layers}
    w = {n: float(m.act_numel) for n, m in layers}
    eff = lo  # all at lo => weighted average lo (weights sum to 1)
    budget_left = max(0.0, target_bit - lo)
    promote = set()
    for n, _ in sorted(layers, key=lambda kv: learned[kv[0]], reverse=True):
        cost = w[n] * (hi - lo)
        if cost <= budget_left + 1e-9:
            promote.add(n); budget_left -= cost; eff += cost

    for n, m in layers:
        with torch.no_grad():
            m.a_bit_int = torch.tensor(hi if n in promote else lo, device=m.a_bit_int.device)
        m._nipq_noise = False
        m.nipq_dynamic_infer = dynamic_infer
        m.a_alpha.requires_grad_(False)
        m.a_bit_param.requires_grad_(False)
    if was_training:
        model.train()
    if verbose:
        print(f"[nipq] done. {len(promote)}/{len(layers)} layers -> A{int(hi)}, rest A{int(lo)}. "
              f"MAC-weighted effective act-bit = {eff:.3f}")
        for n, _ in sorted(layers, key=lambda kv: learned[kv[0]], reverse=True):
            if n in promote:
                print(f"  A{int(hi)}: {n} (learned bit={learned[n]:.2f})")
