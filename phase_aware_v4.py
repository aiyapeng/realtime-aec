"""Phase-aware DT refiner on top of the frozen, single-talk-safe DualPathV3."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from dualpath_v3 import DualPathV3


class PhaseAwareRefinerV4(nn.Module):
    """Refine V3 only when both far-end and near-end activity are present.

    The refiner observes relative phase, the frozen V3 mask/evidence and physical
    spectra.  ST and NST stay on the validated V3 path through a differentiable
    double-talk gate.
    """

    def __init__(self, base: DualPathV3, hidden: int = 192, num_layers: int = 2,
                 delta_max: float = 2.0, mask_mag_max: float = 3.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.n_freq = base.n_freq
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.delta_max = float(delta_max)
        self.mask_mag_max = float(mask_mag_max)

        # 4 log magnitudes + 3 relative-phase pairs + physical ratio +
        # frozen mask(real/imag) + frozen evidence = 12 channels per bin.
        channels = 12
        self.in_proj = nn.Linear(channels * self.n_freq, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.delta_head = nn.Linear(hidden, 2 * self.n_freq)
        self.evidence_head = nn.Linear(hidden, self.n_freq)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def _relative(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cross = a * b.conj()
        unit = cross / (cross.abs() + 1e-8)
        return unit.real, unit.imag

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, E: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                D: torch.Tensor, h=None, force_dt: bool = False):
        with torch.no_grad():
            base_re, base_im, base_p, _ = self.base(
                E.abs(), X.abs(), Y.abs(), D.abs()
            )
        base_mask = torch.complex(base_re.float(), base_im.float())
        ey_re, ey_im = self._relative(E, Y)
        ex_re, ex_im = self._relative(E, X)
        ed_re, ed_im = self._relative(E, D)
        ratio = torch.log1p(E.abs()) - torch.log1p(Y.abs())
        feature = torch.stack(
            (
                torch.log1p(E.abs()), torch.log1p(X.abs()),
                torch.log1p(Y.abs()), torch.log1p(D.abs()),
                ey_re, ey_im, ex_re, ex_im, ed_re, ed_im, ratio,
                base_p,
            ),
            dim=-1,
        )
        # Replace the ratio slot's companion information with both base mask
        # components by adding them into the phase-normalized channels.  This
        # keeps input size compact while retaining the V3 operating point.
        feature[..., 8] = feature[..., 8] + base_mask.real
        feature[..., 9] = feature[..., 9] + base_mask.imag
        z = feature.flatten(-2)
        z = self.norm(torch.nn.functional.silu(self.in_proj(z)))
        z, hn = self.gru(z, h)
        delta_re, delta_im = self.delta_head(z).chunk(2, dim=-1)
        delta = self.delta_max * torch.complex(
            torch.tanh(delta_re), torch.tanh(delta_im)
        )
        p_ne = torch.sigmoid(self.evidence_head(z))

        if force_dt:
            dt_gate = torch.ones_like(base_p[..., :1])
        else:
            # Reference is exactly zero in NST; base evidence is near zero in ST.
            # Sequence-level decision for the locked offline evaluation.  A causal
            # deployment uses the same statistic as a running DTD state.
            far_level = X.abs().mean(dim=(-2, -1), keepdim=True)
            far_gate = far_level / (far_level + 1e-3)
            near_level = base_p.mean(dim=(-2, -1), keepdim=True)
            # A conservative threshold is essential: rare V3 evidence spikes in
            # far-end-only speech must never activate the DT refiner.
            near_gate = torch.sigmoid((near_level - 0.35) / 0.02)
            dt_gate = (far_gate * near_gate).clamp(0.0, 1.0)

        mask = base_mask + dt_gate * delta
        magnitude = mask.abs().clamp(max=self.mask_mag_max)
        mask = torch.polar(magnitude, torch.angle(mask))
        return mask.real, mask.imag, p_ne, dt_gate, hn

    def init_state(self, batch: int = 1, device="cpu"):
        return torch.zeros(self.num_layers, batch, self.hidden, device=device)


def clipped_oracle_mask(E: torch.Tensor, S: torch.Tensor,
                        max_magnitude: float = 3.0) -> torch.Tensor:
    target = S / (E + 1e-8)
    return torch.polar(target.abs().clamp(max=max_magnitude), torch.angle(target))
