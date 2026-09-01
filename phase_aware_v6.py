"""带显式流式状态的复数掩码精修网络。"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualpath_v3 import DualPathV3


class CausalConv2d(nn.Conv2d):
    """Conv2d with left-only time padding and symmetric frequency padding."""

    def __init__(self, in_channels, out_channels, kernel_size, *, dilation=1,
                 bias=True, right_context: int = 0):
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        dil = (dilation, dilation) if isinstance(dilation, int) else dilation
        super().__init__(in_channels, out_channels, kernel, padding=0,
                         dilation=dil, bias=bias)
        total_context = (kernel[0] - 1) * dil[0]
        if not 0 <= right_context <= total_context:
            raise ValueError("right_context exceeds convolution support")
        self.time_future = int(right_context)
        self.time_history = total_context - self.time_future
        self.freq_padding = ((kernel[1] - 1) * dil[1]) // 2

    def forward(self, x):
        x = F.pad(x, (self.freq_padding, self.freq_padding,
                      self.time_history, self.time_future))
        return super().forward(x)


class FrameGroupNorm(nn.Module):
    """Group normalization independent for every time frame.

    Statistics span channels within a group and frequency, never past/future time.
    Weight/bias names and shapes match nn.GroupNorm, allowing V5 initialization.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        batch, channels, time, freq = x.shape
        grouped = x.reshape(batch, self.num_groups,
                            channels // self.num_groups, time, freq)
        mean = grouped.mean(dim=(2, 4), keepdim=True)
        var = grouped.var(dim=(2, 4), unbiased=False, keepdim=True)
        normed = (grouped - mean) * torch.rsqrt(var + self.eps)
        normed = normed.reshape(batch, channels, time, freq)
        return normed * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class CausalResidualTFBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, right_context=(0, 0)):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv2d(channels, channels, 3, dilation=(dilation, 1),
                         right_context=right_context[0]),
            FrameGroupNorm(8, channels),
            nn.SiLU(),
            CausalConv2d(channels, channels, 3, dilation=(dilation, 1),
                         right_context=right_context[1]),
            FrameGroupNorm(8, channels),
        )

    def forward(self, x):
        return F.silu(x + self.net(x))


class PhaseAwareCausalV6(nn.Module):
    """包含 GRU、卷积缓存和场景门控状态的时频精修网络。"""

    def __init__(self, base: DualPathV3, channels: int = 64,
                 delta_max: float = 2.5, mask_mag_max: float = 3.0,
                 fs: int = 16000, hop: int = 128, gate_tau_ms: float = 120.0,
                 near_on: float = 0.22, near_off: float = 0.14,
                 far_threshold: float = 1e-4, confirm_frames: int = 5,
                 hangover_ms: float = 1000.0, lookahead_frames: int = 0,
                 scene_gate_enabled: bool = False, scene_hidden: int = 48,
                 scene_threshold: float = 0.985,
                 scene_low_threshold: float = 0.4,
                 scene_confirm_frames: int = 8,
                 scene_off_threshold: float = 0.01,
                 scene_release_frames: int = 16,
                 scene_soft_gate: bool = False,
                 scene_soft_power: float = 1.0,
                 scene_soft_floor: float = 0.0):
        super().__init__()
        self.base = base
        self.use_aux = True
        self.use_echo = True
        self.use_mic = True
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.delta_max = float(delta_max)
        self.mask_mag_max = float(mask_mag_max)
        self.near_on = float(near_on)
        self.near_off = float(near_off)
        self.far_threshold = float(far_threshold)
        self.confirm_frames = int(confirm_frames)
        self.hangover_frames = max(1, int(round(hangover_ms / (1000.0 * hop / fs))))
        if lookahead_frames not in (0, 4, 8, 16):
            raise ValueError("bounded-lookahead model supports 0, 4, 8 or 16 frames")
        self.lookahead_frames = int(lookahead_frames)
        self.scene_gate_enabled = bool(scene_gate_enabled)
        self.scene_hidden = int(scene_hidden)
        self.scene_threshold = float(scene_threshold)
        self.scene_low_threshold = float(scene_low_threshold)
        self.scene_confirm_frames = int(scene_confirm_frames)
        self.scene_off_threshold = float(scene_off_threshold)
        self.scene_release_frames = int(scene_release_frames)
        self.scene_soft_gate = bool(scene_soft_gate)
        self.scene_soft_power = float(scene_soft_power)
        self.scene_soft_floor = float(scene_soft_floor)
        frame_seconds = hop / fs
        self.gate_alpha = float(math.exp(-frame_seconds / (gate_tau_ms / 1000.0)))

        input_right = 1 if self.lookahead_frames else 0
        if self.lookahead_frames == 4:
            # 1 frame in the input convolution, 2 in the first residual
            # block and 1 in the final unit-dilation block: 4 frames total.
            block_right = ((1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0))
        elif self.lookahead_frames == 8:
            block_right = ((1, 1), (2, 2), (0, 0), (0, 0), (0, 0), (1, 0))
        elif self.lookahead_frames == 16:
            block_right = ((1, 1), (2, 2), (4, 4), (0, 0), (0, 0), (1, 0))
        else:
            block_right = ((0, 0),) * 6
        self.input = nn.Sequential(
            CausalConv2d(14, channels, 3, right_context=input_right),
            FrameGroupNorm(8, channels),
            nn.SiLU(),
        )
        dilations = (1, 2, 4, 8, 16, 1)
        self.blocks = nn.Sequential(*[
            CausalResidualTFBlock(channels, dilation, rights)
            for dilation, rights in zip(dilations, block_right)
        ])
        self.output = nn.Conv2d(channels, 3, 1)
        if self.scene_gate_enabled:
            # Frequency-pooled statistics of all 14 physical/model features.
            # 场景分类头与声学精修网络分离，可单独训练门控参数
            self.scene_norm = nn.LayerNorm(28)
            self.scene_gru = nn.GRU(28, self.scene_hidden, num_layers=2,
                                    batch_first=True, dropout=0.1)
            self.scene_output = nn.Linear(self.scene_hidden, 1)
        # Number of previous feature frames required for exact chunk equivalence.
        total_context = 2 + sum(4 * dilation for dilation in dilations)
        self.history_frames = total_context - self.lookahead_frames

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _cross(a, b):
        value = a * b.conj()
        value = value / (value.abs() + 1e-8)
        return value.real, value.imag

    def _base_and_features(self, E, X, Y, D, h=None):
        with torch.no_grad():
            br, bi, bp, hn = self.base(E.abs(), X.abs(), Y.abs(), D.abs(), h)
        base = torch.complex(br.float(), bi.float())
        ey = self._cross(E, Y)
        ex = self._cross(E, X)
        ed = self._cross(E, D)
        features = torch.stack((
            torch.log1p(E.abs()), torch.log1p(X.abs()), torch.log1p(Y.abs()),
            torch.log1p(D.abs()), ey[0], ey[1], ex[0], ex[1], ed[0], ed[1],
            torch.log1p(E.abs()) - torch.log1p(Y.abs()), br, bi, bp,
        ), dim=1)
        return base, bp, features, hn

    def _refine(self, features):
        raw = self.output(self.blocks(self.input(features)))
        delta = self.delta_max * torch.complex(torch.tanh(raw[:, 0]),
                                               torch.tanh(raw[:, 1]))
        p_ne = torch.sigmoid(raw[:, 2])
        return delta, p_ne

    @staticmethod
    def _init_conv_stream_state(conv, batch, device, dtype, n_freq):
        return {
            "buffer": torch.empty(batch, conv.in_channels, 0, n_freq,
                                  device=device, dtype=dtype),
            "seen": 0,
            "emitted": 0,
        }

    @staticmethod
    def _conv_stream_push(conv, current, state, *, final=False):
        """Emit only newly ready frames of one time-causal/lookahead conv.

        The cache is local to this convolutional layer.  This avoids running
        the complete 126-frame network receptive field again for every small
        input chunk while preserving the exact zero-boundary convention.
        """
        combined = torch.cat((state["buffer"], current), dim=2)
        previous_seen = int(state["seen"])
        buffer_start = previous_seen - state["buffer"].shape[2]
        seen = previous_seen + current.shape[2]
        emitted = int(state["emitted"])
        ready = seen if final else max(0, seen - conv.time_future)
        count = ready - emitted
        if count > 0:
            required_start = emitted - conv.time_history
            required_end = ready + conv.time_future
            source_start = max(0, required_start)
            source_end = min(seen, required_end)
            local_start = source_start - buffer_start
            local_end = source_end - buffer_start
            source = combined[:, :, local_start:local_end]
            source = F.pad(
                source,
                (conv.freq_padding, conv.freq_padding,
                 source_start - required_start, required_end - source_end),
            )
            output = F.conv2d(source, conv.weight, conv.bias,
                              dilation=conv.dilation)
            if output.shape[2] != count:
                raise RuntimeError("incremental convolution frame mismatch")
        else:
            output = current.new_empty(
                current.shape[0], conv.out_channels, 0, current.shape[3])
        total_context = conv.time_history + conv.time_future
        if total_context:
            buffer = combined[:, :, -total_context:].detach()
        else:
            buffer = combined[:, :, :0].detach()
        return output, {"buffer": buffer, "seen": seen, "emitted": ready}

    def _init_refiner_stream_state(self, batch, device, dtype):
        n_freq = self.base.n_freq
        state = {
            "input": self._init_conv_stream_state(
                self.input[0], batch, device, dtype, n_freq),
            "blocks": [],
        }
        for block in self.blocks:
            state["blocks"].append({
                "conv1": self._init_conv_stream_state(
                    block.net[0], batch, device, dtype, n_freq),
                "conv2": self._init_conv_stream_state(
                    block.net[3], batch, device, dtype, n_freq),
                "residual": torch.empty(
                    batch, block.net[0].in_channels, 0, n_freq,
                    device=device, dtype=dtype),
            })
        return state

    def _refine_stream_push(self, current, state, *, final=False):
        value, input_state = self._conv_stream_push(
            self.input[0], current, state["input"], final=final)
        if value.shape[2]:
            value = self.input[2](self.input[1](value))
        block_states = []
        for block, old_state in zip(self.blocks, state["blocks"]):
            residual = torch.cat((old_state["residual"], value), dim=2)
            hidden, conv1_state = self._conv_stream_push(
                block.net[0], value, old_state["conv1"], final=final)
            if hidden.shape[2]:
                hidden = block.net[2](block.net[1](hidden))
            hidden, conv2_state = self._conv_stream_push(
                block.net[3], hidden, old_state["conv2"], final=final)
            if hidden.shape[2]:
                hidden = block.net[4](hidden)
                count = hidden.shape[2]
                if residual.shape[2] < count:
                    raise RuntimeError("incremental residual frame mismatch")
                value = F.silu(residual[:, :, :count] + hidden)
                residual = residual[:, :, count:]
            else:
                value = hidden
            block_states.append({
                "conv1": conv1_state,
                "conv2": conv2_state,
                "residual": residual.detach(),
            })
        if value.shape[2]:
            raw = self.output(value)
            delta = self.delta_max * torch.complex(torch.tanh(raw[:, 0]),
                                                   torch.tanh(raw[:, 1]))
            p_ne = torch.sigmoid(raw[:, 2])
        else:
            delta = torch.empty(current.shape[0], 0, current.shape[3],
                                device=current.device, dtype=torch.complex64)
            p_ne = current.new_empty(current.shape[0], 0, current.shape[3])
        return delta, p_ne, {"input": input_state, "blocks": block_states}

    def _scene_logits(self, features, h=None):
        if not self.scene_gate_enabled:
            raise RuntimeError("the learned scene gate is not enabled")
        mean = features.mean(dim=-1).transpose(1, 2)
        rms = features.square().mean(dim=-1).add(1e-8).sqrt().transpose(1, 2)
        pooled = self.scene_norm(torch.cat((mean, rms), dim=-1))
        recurrent, hn = self.scene_gru(pooled, h)
        return self.scene_output(recurrent), hn

    def _scene_gate_full(self, features):
        logits, _ = self._scene_logits(features)
        frames = logits.shape[1]
        indices = (torch.arange(frames, device=logits.device) +
                   self.lookahead_frames).clamp(max=frames - 1)
        aligned = torch.sigmoid(logits.index_select(1, indices))
        if self.scene_soft_gate:
            return self._soft_scene_gate(aligned)
        return self._scene_hysteresis(aligned)[0]

    def _soft_scene_gate(self, probabilities):
        if self.scene_soft_floor > 0:
            scaled = ((probabilities - self.scene_soft_floor) /
                      (1.0 - self.scene_soft_floor)).clamp(0.0, 1.0)
            probabilities = scaled.square() * (3.0 - 2.0 * scaled)
        return probabilities.pow(self.scene_soft_power)

    def _scene_hysteresis(self, probabilities, active=None, on_count=None,
                          off_count=None):
        batch = probabilities.shape[0]
        if active is None:
            active = torch.zeros(batch, device=probabilities.device,
                                 dtype=torch.bool)
        if off_count is None:
            off_count = torch.zeros(batch, device=probabilities.device,
                                    dtype=torch.long)
        if on_count is None:
            on_count = torch.zeros(batch, device=probabilities.device,
                                   dtype=torch.long)
        gates = []
        for index in range(probabilities.shape[1]):
            value = probabilities[:, index, 0]
            low = value >= self.scene_low_threshold
            on_count = torch.where((~active) & low, on_count + 1,
                                   torch.zeros_like(on_count))
            turn_on = (~active) & ((value >= self.scene_threshold) |
                                   (on_count >= self.scene_confirm_frames))
            active = active | turn_on
            on_count = torch.where(active, torch.zeros_like(on_count), on_count)
            release = active & (value < self.scene_off_threshold)
            off_count = torch.where(release, off_count + 1,
                                    torch.zeros_like(off_count))
            turn_off = active & (off_count >= self.scene_release_frames)
            active = active & ~turn_off
            off_count = torch.where(active, off_count,
                                    torch.zeros_like(off_count))
            gates.append(active.to(probabilities.dtype))
        gate = torch.stack(gates, dim=1).unsqueeze(-1)
        return gate, active.detach(), on_count.detach(), off_count.detach()

    def _gate(self, X, bp, state: Optional[Dict[str, torch.Tensor]] = None):
        batch, time, _ = bp.shape
        if state is None:
            far_ema = torch.zeros(batch, device=bp.device, dtype=bp.dtype)
            near_ema = torch.zeros_like(far_ema)
            active = torch.zeros(batch, device=bp.device, dtype=torch.bool)
            on_count = torch.zeros(batch, device=bp.device, dtype=torch.long)
            hangover = torch.zeros_like(on_count)
        else:
            far_ema = state["far_ema"]
            near_ema = state["near_ema"]
            active = state["gate_active"]
            on_count = state["on_count"]
            hangover = state["hangover"]
        far_frames = X.abs().mean(dim=-1)
        near_frames = bp.mean(dim=-1)
        gates = []
        alpha = self.gate_alpha
        for index in range(time):
            far_ema = alpha * far_ema + (1.0 - alpha) * far_frames[:, index]
            near_ema = alpha * near_ema + (1.0 - alpha) * near_frames[:, index]
            far_active = far_ema > self.far_threshold
            candidate = far_active & (near_ema > self.near_on)
            on_count = torch.where(candidate, on_count + 1,
                                   torch.zeros_like(on_count))
            turn_on = on_count >= self.confirm_frames
            near_present = far_active & (near_ema > self.near_off)
            refreshed = torch.full_like(hangover, self.hangover_frames)
            hangover = torch.where(active & near_present, refreshed,
                                   torch.clamp(hangover - 1, min=0))
            active = far_active & (turn_on | near_present & active |
                                   ((hangover > 0) & active))
            hangover = torch.where(turn_on, refreshed, hangover)
            gates.append(active.to(bp.dtype))
        gate = torch.stack(gates, dim=1).unsqueeze(-1)
        return gate, {"far_ema": far_ema.detach(),
                      "near_ema": near_ema.detach(),
                      "gate_active": active.detach(),
                      "on_count": on_count.detach(),
                      "hangover": hangover.detach()}

    def _mask(self, base, delta, gate):
        mask = base + gate * delta
        return torch.polar(mask.abs().clamp(max=self.mask_mag_max), torch.angle(mask))

    def forward(self, E, X, Y, D, force_dt: bool = False):
        base, bp, features, _ = self._base_and_features(E, X, Y, D)
        delta, p_ne = self._refine(features)
        if force_dt:
            gate = torch.ones_like(bp[..., :1])
        elif self.scene_gate_enabled:
            gate = self._scene_gate_full(features)
        else:
            gate = self._gate(X, bp)[0]
        mask = self._mask(base, delta, gate)
        return mask.real, mask.imag, p_ne, gate

    def init_stream_state(self, batch: int = 1, device="cpu"):
        dtype = next(self.parameters()).dtype
        state = {
            "base_h": self.base.init_state(batch, device),
            "feature_cache": torch.empty(batch, 14, 0, self.base.n_freq,
                                         device=device, dtype=dtype),
            "far_ema": torch.zeros(batch, device=device, dtype=dtype),
            "near_ema": torch.zeros(batch, device=device, dtype=dtype),
            "gate_active": torch.zeros(batch, device=device, dtype=torch.bool),
            "on_count": torch.zeros(batch, device=device, dtype=torch.long),
            "hangover": torch.zeros(batch, device=device, dtype=torch.long),
        }
        if self.scene_gate_enabled:
            state["scene_h"] = torch.zeros(2, batch, self.scene_hidden,
                                             device=device, dtype=dtype)
            state["scene_active"] = torch.zeros(batch, device=device,
                                                  dtype=torch.bool)
            state["scene_on_count"] = torch.zeros(batch, device=device,
                                                    dtype=torch.long)
            state["scene_off_count"] = torch.zeros(batch, device=device,
                                                     dtype=torch.long)
        return state

    def stream_step(self, E, X, Y, D, state, force_dt: bool = False):
        """Process one or more new frames and return updated explicit state."""
        if self.lookahead_frames:
            raise RuntimeError("use the bounded-lookahead runner for lookahead V6")
        base, bp, current, hn = self._base_and_features(
            E, X, Y, D, state["base_h"])
        cached = state["feature_cache"]
        combined = torch.cat((cached, current), dim=2)
        delta_all, p_all = self._refine(combined)
        frames = current.shape[2]
        delta = delta_all[:, -frames:]
        p_ne = p_all[:, -frames:]
        if force_dt:
            gate = torch.ones_like(bp[..., :1])
            gate_state = {key: state[key] for key in
                          ("far_ema", "near_ema", "gate_active",
                           "on_count", "hangover")}
        elif self.scene_gate_enabled:
            logits, scene_h = self._scene_logits(current, state["scene_h"])
            probabilities = torch.sigmoid(logits)
            if self.scene_soft_gate:
                gate = self._soft_scene_gate(probabilities)
                scene_active = state["scene_active"]
                scene_on_count = state["scene_on_count"]
                scene_off_count = state["scene_off_count"]
            else:
                gate, scene_active, scene_on_count, scene_off_count = self._scene_hysteresis(
                    probabilities, state["scene_active"],
                    state["scene_on_count"],
                    state["scene_off_count"])
            gate_state = {key: state[key] for key in
                          ("far_ema", "near_ema", "gate_active",
                           "on_count", "hangover")}
        else:
            gate, gate_state = self._gate(X, bp, state)
        mask = self._mask(base, delta, gate)
        new_state = {
            "base_h": hn.detach(),
            "feature_cache": combined[:, :, -self.history_frames:].detach(),
            **gate_state,
        }
        if self.scene_gate_enabled:
            if force_dt:
                _, scene_h = self._scene_logits(current, state["scene_h"])
                scene_active = state["scene_active"]
                scene_on_count = state["scene_on_count"]
                scene_off_count = state["scene_off_count"]
            new_state["scene_h"] = scene_h.detach()
            new_state["scene_active"] = scene_active.detach()
            new_state["scene_on_count"] = scene_on_count.detach()
            new_state["scene_off_count"] = scene_off_count.detach()
        return mask.real, mask.imag, p_ne, gate, new_state

    def init_lookahead_state(self, batch: int = 1, device="cpu"):
        """State for the exact fixed-lookahead chunk runner."""
        if not self.lookahead_frames:
            raise RuntimeError("lookahead state is only valid when lookahead_frames > 0")
        dtype = next(self.parameters()).dtype
        empty_features = torch.empty(batch, 14, 0, self.base.n_freq,
                                     device=device, dtype=dtype)
        empty_complex = torch.empty(batch, 0, self.base.n_freq,
                                    device=device, dtype=torch.complex64)
        empty_gate = torch.empty(batch, 0, 1, device=device, dtype=dtype)
        state = {
            "base_h": self.base.init_state(batch, device),
            "features": empty_features,
            "bases": empty_complex,
            "gates": empty_gate,
            "far_ema": torch.zeros(batch, device=device, dtype=dtype),
            "near_ema": torch.zeros(batch, device=device, dtype=dtype),
            "gate_active": torch.zeros(batch, device=device, dtype=torch.bool),
            "on_count": torch.zeros(batch, device=device, dtype=torch.long),
            "hangover": torch.zeros(batch, device=device, dtype=torch.long),
            "buffer_start": 0,
            "seen": 0,
            "emitted": 0,
        }
        if self.scene_gate_enabled:
            state["scene_h"] = torch.zeros(2, batch, self.scene_hidden,
                                             device=device, dtype=dtype)
            state["scene_probs"] = empty_gate
            state["scene_active"] = torch.zeros(batch, device=device,
                                                  dtype=torch.bool)
            state["scene_on_count"] = torch.zeros(batch, device=device,
                                                    dtype=torch.long)
            state["scene_off_count"] = torch.zeros(batch, device=device,
                                                     dtype=torch.long)
        return state

    def init_incremental_lookahead_state(self, batch: int = 1, device="cpu"):
        """Lookahead state with exact layer-local convolution caches."""
        state = self.init_lookahead_state(batch=batch, device=device)
        dtype = next(self.parameters()).dtype
        state["refiner_stream"] = self._init_refiner_stream_state(
            batch, device, dtype)
        return state

    def lookahead_push(self, E, X, Y, D, state, *, final: bool = False,
                       force_dt: bool = False):
        """Push new frames and emit every mask whose fixed future is available.

        With ``final=True`` the remaining tail is emitted using the same zero-right
        boundary condition as full-sequence inference.
        """
        frames = E.shape[1]
        incremental_refiner = "refiner_stream" in state
        if frames:
            base, bp, current, hn = self._base_and_features(
                E, X, Y, D, state["base_h"])
            if self.scene_gate_enabled:
                scene_logits, scene_h = self._scene_logits(current, state["scene_h"])
                scene_probs = torch.cat((state["scene_probs"],
                                         torch.sigmoid(scene_logits)), dim=1)
            if force_dt:
                gate = torch.ones_like(bp[..., :1])
                gate_state = {key: state[key] for key in
                              ("far_ema", "near_ema", "gate_active",
                               "on_count", "hangover")}
            elif self.scene_gate_enabled:
                gate = torch.empty_like(bp[..., :1])
                gate_state = {key: state[key] for key in
                              ("far_ema", "near_ema", "gate_active",
                               "on_count", "hangover")}
            else:
                gate, gate_state = self._gate(X, bp, state)
            features = (current if incremental_refiner else
                        torch.cat((state["features"], current), dim=2))
            bases = torch.cat((state["bases"], base), dim=1)
            gates = torch.cat((state["gates"], gate), dim=1)
            seen = state["seen"] + frames
        else:
            hn = state["base_h"]
            gate_state = {key: state[key] for key in
                          ("far_ema", "near_ema", "gate_active",
                           "on_count", "hangover")}
            features, bases, gates = state["features"], state["bases"], state["gates"]
            if self.scene_gate_enabled:
                scene_h, scene_probs = state["scene_h"], state["scene_probs"]
            seen = state["seen"]
            current = state["features"][:, :, :0]

        ready = seen if final else max(0, seen - self.lookahead_frames)
        first = state["emitted"] - state["buffer_start"]
        last = ready - state["buffer_start"]
        if last > first:
            if incremental_refiner:
                delta, p_ne, refiner_stream = self._refine_stream_push(
                    current, state["refiner_stream"], final=final)
                if delta.shape[1] != last - first:
                    raise RuntimeError("incremental refiner readiness mismatch")
            else:
                delta_all, p_all = self._refine(features)
                delta = delta_all[:, first:last]
                p_ne = p_all[:, first:last]
            if self.scene_gate_enabled and not force_dt:
                target_global = torch.arange(state["emitted"], ready,
                                             device=features.device)
                evidence_global = (target_global + self.lookahead_frames).clamp(
                    max=max(0, seen - 1))
                evidence_local = evidence_global - state["buffer_start"]
                aligned_probs = scene_probs.index_select(1, evidence_local)
                if self.scene_soft_gate:
                    gate_out = self._soft_scene_gate(aligned_probs)
                    scene_active = state["scene_active"]
                    scene_on_count = state["scene_on_count"]
                    scene_off_count = state["scene_off_count"]
                else:
                    gate_out, scene_active, scene_on_count, scene_off_count = self._scene_hysteresis(
                        aligned_probs, state["scene_active"],
                        state["scene_on_count"],
                        state["scene_off_count"])
            else:
                gate_out = gates[:, first:last]
            base_out = bases[:, first:last]
            mask = self._mask(base_out, delta, gate_out)
        else:
            if incremental_refiner:
                delta, p_ne, refiner_stream = self._refine_stream_push(
                    current, state["refiner_stream"], final=final)
                if delta.shape[1]:
                    raise RuntimeError("unexpected incremental refiner output")
            batch = features.shape[0]
            mask = torch.empty(batch, 0, self.base.n_freq, device=features.device,
                               dtype=torch.complex64)
            p_ne = torch.empty(batch, 0, self.base.n_freq, device=features.device,
                               dtype=features.dtype)
            gate_out = torch.empty(batch, 0, 1, device=features.device,
                                   dtype=features.dtype)

        emitted = ready
        keep_from = max(0, emitted - self.history_frames)
        trim = keep_from - state["buffer_start"]
        if trim > 0:
            if not incremental_refiner:
                features = features[:, :, trim:]
            bases = bases[:, trim:]
            gates = gates[:, trim:]
            if self.scene_gate_enabled:
                scene_probs = scene_probs[:, trim:]
        new_state = {
            "base_h": hn.detach(),
            "features": features.detach(),
            "bases": bases.detach(),
            "gates": gates.detach(),
            **gate_state,
            "buffer_start": keep_from,
            "seen": seen,
            "emitted": emitted,
        }
        if incremental_refiner:
            new_state["features"] = state["features"][:, :, :0]
            new_state["refiner_stream"] = refiner_stream
        if self.scene_gate_enabled:
            new_state["scene_h"] = scene_h.detach()
            new_state["scene_probs"] = scene_probs.detach()
            if force_dt or last <= first:
                scene_active = state["scene_active"]
                scene_on_count = state["scene_on_count"]
                scene_off_count = state["scene_off_count"]
            new_state["scene_active"] = scene_active.detach()
            new_state["scene_on_count"] = scene_on_count.detach()
            new_state["scene_off_count"] = scene_off_count.detach()
        return mask.real, mask.imag, p_ne, gate_out, new_state
