"""带近端保护下限的因果 GRU 复数掩码估计器。"""

import torch
import torch.nn as nn


class DualPathV3(nn.Module):
    def __init__(self, n_freq=193, hidden=256, num_layers=3,
                 mask_mag_max=3.0, use_mic=True, phase_max=1.0471975512):
        super().__init__()
        self.n_freq = int(n_freq)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.mask_mag_max = float(mask_mag_max)
        self.phase_max = float(phase_max)
        self.use_aux = True
        self.use_echo = True
        self.use_mic = bool(use_mic)

        n_ch = 3 + (1 if self.use_mic else 0)
        self.in_proj = nn.Linear(n_freq * n_ch, hidden)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)

        self.ev = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                nn.Linear(hidden // 2, n_freq))
        self.film = nn.Linear(n_freq, 2 * hidden)
        self.mask_head = nn.Linear(hidden, 3 * n_freq)     # mag gate, dir re, dir im
        # learned near-end floor scalar (sigmoid -> ~0.6 at init)
        self.floor_logit = nn.Parameter(torch.tensor(0.405))
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            b = self.mask_head.bias.view(3, self.n_freq)
            p = 1.0 / self.mask_mag_max
            b[0].fill_(float(torch.log(torch.tensor(p / (1 - p)))))
            b[1].fill_(1.0); b[2].fill_(0.0)
            self.mask_head.weight.data[:self.n_freq].mul_(0.1)
            self.film.weight.data.mul_(0.1); self.film.bias.data.zero_()

    def _feature(self, me, mx, my, md=None):
        f = [torch.log1p(me), torch.log1p(mx), torch.log1p(my)]
        if self.use_mic:
            assert md is not None
            f.append(torch.log1p(md))
        return torch.cat(f, dim=-1)

    def forward(self, mag_err, mag_ref, mag_echo, mag_mic=None, h=None):
        feat = self._feature(mag_err, mag_ref, mag_echo, mag_mic)
        z = self.norm(self.act(self.in_proj(feat)))
        z, hn = self.gru(z, h)

        p_ne = torch.sigmoid(self.ev(z))                         # (B,T,F)
        gamma, beta = self.film(p_ne).chunk(2, dim=-1)
        zc = z * (1.0 + torch.tanh(gamma)) + beta

        mag_l, dr, di = self.mask_head(zc).chunk(3, dim=-1)
        floor = torch.sigmoid(self.floor_logit) * p_ne           # near-end floor
        mag = floor + (self.mask_mag_max - floor) * torch.sigmoid(mag_l)
        denom = torch.sqrt(dr * dr + di * di + 1e-8)
        mask_re = mag * dr / denom
        mask_im = mag * di / denom
        return mask_re, mask_im, p_ne, hn

    def init_state(self, batch=1, device="cpu"):
        return torch.zeros(self.num_layers, batch, self.hidden, device=device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
