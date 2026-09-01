"""基于因果 GRU 的残余回声与噪声后滤波器。"""

import torch
import torch.nn as nn


class ResSuppressor(nn.Module):
    def __init__(self, n_freq=257, hidden=256, num_layers=2, use_aux=True,
                 mask_floor=0.0, use_echo=False):
        super().__init__()
        self.n_freq = int(n_freq)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.use_aux = bool(use_aux)
        self.use_echo = bool(use_echo)         # 额外输入线性回声估计|Ŷ|, 用于回声-近端解耦
        self.mask_floor = float(mask_floor)    # 增益下限, 防深零点/音乐噪声(0=无下限)

        n_ch = 1 + (1 if use_aux else 0) + (1 if use_echo else 0)
        in_dim = n_freq * n_ch
        self.in_proj = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_freq)

    def make_feature(self, mag_err, mag_ref=None, mag_echo=None):
        feats = [torch.log1p(mag_err)]
        if self.use_aux:
            assert mag_ref is not None
            feats.append(torch.log1p(mag_ref))
        if self.use_echo:
            assert mag_echo is not None
            feats.append(torch.log1p(mag_echo))
        return torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

    def forward(self, mag_err, mag_ref=None, mag_echo=None, h=None):
        """mag_err/mag_ref/mag_echo: (B,T,F)。返回 (mask (B,T,F), h_n)。
        h 为 None 时 GRU 自动置零初值；流式时逐帧传入/传出。
        mag_echo: 线性AEC的回声估计幅度(use_echo=True 时必填), 帮助区分残余回声与近端。
        mask ∈ [mask_floor, 1]: 下限抬起避免深零点产生音乐噪声。"""
        feat = self.make_feature(mag_err, mag_ref, mag_echo)
        z = self.norm(self.act(self.in_proj(feat)))
        z, hn = self.gru(z, h)
        g = torch.sigmoid(self.out(z))
        mask = self.mask_floor + (1.0 - self.mask_floor) * g
        return mask, hn

    def init_state(self, batch=1, device="cpu"):
        return torch.zeros(self.num_layers, batch, self.hidden, device=device)


class ScenarioDualPathSuppressor(nn.Module):
    """Causal scenario-conditioned post-filter with separated responsibilities.

    The recurrent echo path estimates aggressive suppression from residual,
    far-reference and MDF echo evidence.  A separate frame-local speech path
    estimates near-end dominance and a conservative speech-recovery mask.  It
    has no shared trainable trunk with the echo GRU, so near-end supervision
    cannot be overwritten by the far-end suppression objective.  A physical
    far-activity gate enforces the NEST identity path.

    Only the echo path carries recurrent state, preserving the original single
    ``h_in/h_out`` streaming and ONNX contract.
    """
    model_type = "scenario_dual_path_v1"

    def __init__(self, n_freq=193, hidden=256, num_layers=3, use_aux=True,
                 mask_floor=0.05, use_echo=True, speech_hidden=128):
        super().__init__()
        self.n_freq = int(n_freq)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.use_aux = bool(use_aux)
        self.use_echo = bool(use_echo)
        self.mask_floor = float(mask_floor)
        self.speech_hidden = int(speech_hidden)

        echo_ch = 1 + (1 if self.use_aux else 0) + (1 if self.use_echo else 0)
        self.echo_in = nn.Linear(self.n_freq * echo_ch, self.hidden)
        self.echo_norm = nn.LayerNorm(self.hidden)
        self.echo_gru = nn.GRU(self.hidden, self.hidden, num_layers=self.num_layers,
                               batch_first=True)
        self.echo_out = nn.Linear(self.hidden, self.n_freq)

        # Independent near-end evidence/recovery trunk.  The normalized
        # residual-to-echo ratio is included because absolute level alone is
        # ambiguous in real double talk.
        speech_ch = 4
        self.speech_in = nn.Linear(self.n_freq * speech_ch, self.speech_hidden)
        self.speech_norm = nn.LayerNorm(self.speech_hidden)
        self.speech_mid = nn.Linear(self.speech_hidden, self.speech_hidden)
        self.speech_prob_out = nn.Linear(self.speech_hidden, self.n_freq)
        self.speech_mask_out = nn.Linear(self.speech_hidden, self.n_freq)
        self.act = nn.SiLU()

        with torch.no_grad():
            self.echo_out.bias.fill_(1.5)       # stable, initially permissive
            self.speech_mask_out.bias.fill_(3.0)  # NEST starts near identity
            self.speech_prob_out.bias.fill_(-1.0)

    def _echo_feature(self, mag_err, mag_ref, mag_echo):
        feats = [torch.log1p(mag_err)]
        if self.use_aux:
            if mag_ref is None:
                raise ValueError("mag_ref is required")
            feats.append(torch.log1p(mag_ref))
        if self.use_echo:
            if mag_echo is None:
                raise ValueError("mag_echo is required")
            feats.append(torch.log1p(mag_echo))
        return torch.cat(feats, dim=-1)

    def forward(self, mag_err, mag_ref=None, mag_echo=None, h=None,
                return_aux=False):
        eps = 1e-6
        if mag_ref is None:
            mag_ref = torch.zeros_like(mag_err)
        if mag_echo is None:
            mag_echo = torch.zeros_like(mag_err)

        ez = self.echo_norm(self.act(self.echo_in(
            self._echo_feature(mag_err, mag_ref, mag_echo))))
        ez, h_out = self.echo_gru(ez, h)
        echo_mask = self.mask_floor + (1.0 - self.mask_floor) * torch.sigmoid(
            self.echo_out(ez))

        ratio = torch.log((mag_err + eps) / (mag_echo + eps)).clamp(-8.0, 8.0)
        speech_feat = torch.cat([
            torch.log1p(mag_err), torch.log1p(mag_ref),
            torch.log1p(mag_echo), ratio,
        ], dim=-1)
        sz = self.speech_norm(self.act(self.speech_in(speech_feat)))
        sz = self.act(self.speech_mid(sz))
        speech_logits = self.speech_prob_out(sz)
        speech_prob = torch.sigmoid(speech_logits)
        speech_mask = self.mask_floor + (1.0 - self.mask_floor) * torch.sigmoid(
            self.speech_mask_out(sz))

        # Purely physical far activity: exactly zero for NEST and bounded for
        # arbitrary recording gain.  It is intentionally not trainable.
        far_level = mag_ref.mean(dim=-1, keepdim=True)
        residual_level = mag_err.mean(dim=-1, keepdim=True)
        far_gate = (far_level / (far_level + residual_level + eps)).detach()
        dt_mask = speech_prob * speech_mask + (1.0 - speech_prob) * echo_mask
        mask = (1.0 - far_gate) * speech_mask + far_gate * dt_mask

        if return_aux:
            aux = {"echo_mask": echo_mask, "speech_mask": speech_mask,
                   "speech_prob": speech_prob, "speech_logits": speech_logits,
                   "far_gate": far_gate}
            return mask, h_out, aux
        return mask, h_out

    def init_state(self, batch=1, device="cpu"):
        return torch.zeros(self.num_layers, batch, self.hidden, device=device)


# ------------------------- 掩蔽应用 & 损失 -------------------------

def apply_mask(mask, err_complex):
    """把实数掩蔽作用到误差复谱上（幅度乘掩蔽, 相位不变）。
    mask:(B,T,F) 实; err_complex:(B,T,F) 复。返回增强后的复谱。"""
    return err_complex * mask.to(err_complex.real.dtype)


def _compress_complex(spec, c=0.3, eps=1e-8):
    mag = torch.abs(spec)
    comp_mag = (mag + eps) ** c
    phase = spec / (mag + eps)
    return comp_mag, comp_mag * phase   # (压缩幅度, 压缩复谱)


def compressed_loss(enh_complex, tgt_complex, c=0.3, alpha=0.3):
    """压缩谱损失(Braun & Tashev 风格)：幅度项 + 复数项加权。
    对回声/噪声抑制稳定有效。"""
    m_e, s_e = _compress_complex(enh_complex, c)
    m_t, s_t = _compress_complex(tgt_complex, c)
    loss_mag = torch.mean((m_e - m_t) ** 2)
    loss_cplx = torch.mean(torch.abs(s_e - s_t) ** 2)
    return (1 - alpha) * loss_mag + alpha * loss_cplx


def si_sdr_loss(est, ref, eps=1e-8):
    """时域 SI-SDR 损失(取负)。est/ref:(B, samples)。"""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    alpha = (torch.sum(est * ref, dim=-1, keepdim=True) /
             (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps))
    target = alpha * ref
    noise = est - target
    si_sdr = 10 * torch.log10(
        (torch.sum(target ** 2, dim=-1) + eps) /
        (torch.sum(noise ** 2, dim=-1) + eps))
    return -si_sdr.mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
