"""Time-frequency preserving convolutional complex-mask DT refiner."""
from __future__ import annotations

import torch
import torch.nn as nn

from dualpath_v3 import DualPathV3


class ResidualTFBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=(dilation, 1),
                      dilation=(dilation, 1)),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=(dilation, 1),
                      dilation=(dilation, 1)),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))


class PhaseAwareConvV5(nn.Module):
    def __init__(self, base: DualPathV3, channels: int = 64,
                 delta_max: float = 2.5, mask_mag_max: float = 3.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.delta_max = float(delta_max)
        self.mask_mag_max = float(mask_mag_max)
        self.input = nn.Sequential(nn.Conv2d(14, channels, 3, padding=1),
                                   nn.GroupNorm(8, channels), nn.SiLU())
        self.blocks = nn.Sequential(*[
            ResidualTFBlock(channels, dilation)
            for dilation in (1, 2, 4, 8, 16, 1)
        ])
        self.output = nn.Conv2d(channels, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _cross(a, b):
        value = a * b.conj()
        value = value / (value.abs() + 1e-8)
        return value.real, value.imag

    def forward(self, E, X, Y, D, force_dt: bool = False):
        with torch.no_grad():
            br, bi, bp, _ = self.base(E.abs(), X.abs(), Y.abs(), D.abs())
        base = torch.complex(br.float(), bi.float())
        ey = self._cross(E, Y)
        ex = self._cross(E, X)
        ed = self._cross(E, D)
        features = torch.stack((
            torch.log1p(E.abs()), torch.log1p(X.abs()), torch.log1p(Y.abs()),
            torch.log1p(D.abs()), ey[0], ey[1], ex[0], ex[1], ed[0], ed[1],
            torch.log1p(E.abs()) - torch.log1p(Y.abs()),
            br, bi, bp,
        ), dim=1)
        z = self.blocks(self.input(features))
        raw = self.output(z)
        delta = self.delta_max * torch.complex(torch.tanh(raw[:, 0]),
                                               torch.tanh(raw[:, 1]))
        p_ne = torch.sigmoid(raw[:, 2])
        if force_dt:
            gate = torch.ones_like(bp[..., :1])
        else:
            far = X.abs().mean(dim=(-2, -1), keepdim=True)
            near = bp.mean(dim=(-2, -1), keepdim=True)
            # Physical scenario gate, using only signals available at inference:
            # NST has no far reference; ST has essentially zero V3 near evidence;
            # DT has both.  A hard decision avoids partially mixing two separately
            # trained operating points, which was the final source of DT damage.
            # Locked dev separation: ST max=0.105, DT min=0.253.  The midpoint
            # threshold leaves margin on both sides and prevents ST false positives.
            gate = ((far > 1e-4) & (near > 0.18)).to(bp.dtype)
        mask = base + gate * delta
        mask = torch.polar(mask.abs().clamp(max=self.mask_mag_max), torch.angle(mask))
        return mask.real, mask.imag, p_ne, gate
