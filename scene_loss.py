"""按近端主导度加权的回声抑制与近端保真损失。"""

import torch
import torch.nn.functional as F


def _compress(x, c=0.3, eps=1e-8):
    """压缩谱（Braun & Tashev 风格）。返回 (压缩幅度, 压缩复谱)。"""
    mag = x.abs()
    cm = (mag + eps) ** c
    return cm, cm * (x / (mag + eps))


def near_dominance_target(E, S, eps=1e-8):
    """逐时频近端主导度 |S|²/(|S|²+|E-S|²) ∈ [0,1]，(B,T,F) 实数。detach 后当标签用。"""
    ps = S.abs() ** 2
    pn = (E - S).abs() ** 2
    return (ps / (ps + pn + eps)).clamp(0.0, 1.0)


def scene_loss(enh, E, S, p_ne, *, c=0.3, alpha=0.3,
               w_near=3.0, w_echo=1.0, w_evi=0.5, eps=1e-8):
    """场景门控损失。返回 (total, parts_dict)。

    l_spec : 基础压缩谱映射损失 enh->S（幅度项 + 复数项）。
    l_near : 近端主导加权的压缩幅度距离。
    l_echo : 回声主导加权的输出能量。
    l_evi  : 证据分支 BCE（软标签 = 近端主导度）。
    """
    cm_e, cc_e = _compress(enh, c)
    cm_s, cc_s = _compress(S, c)
    l_mag = ((cm_e - cm_s) ** 2).mean()
    l_cplx = ((cc_e - cc_s).abs() ** 2).mean()
    l_spec = (1 - alpha) * l_mag + alpha * l_cplx

    d = near_dominance_target(E, S).detach()          # (B,T,F) 软标签
    per_tf = (cm_e - cm_s) ** 2
    l_near = (d * per_tf).sum() / (d.sum() + eps)     # 近端主导区域保真
    l_echo = ((1 - d) * (cm_e ** 2)).sum() / ((1 - d).sum() + eps)  # 回声主导区域抑制

    l_evi = F.binary_cross_entropy(p_ne.clamp(1e-6, 1 - 1e-6), d)

    total = l_spec + w_near * l_near + w_echo * l_echo + w_evi * l_evi
    parts = {"spec": float(l_spec.detach()), "near": float(l_near.detach()),
             "echo": float(l_echo.detach()), "evi": float(l_evi.detach())}
    return total, parts


def near_restricted_tv(mask_mag, S_mag, E_mag, eps=1e-6):
    """在近端主导区域约束掩码时频平滑度。输入形状为 (B,T,F)。"""
    w = S_mag / (S_mag + E_mag + eps)                 # 近端主导度∈[0,1]
    wt = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
    wf = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
    tv_t = (wt * (mask_mag[:, 1:, :] - mask_mag[:, :-1, :]).abs()).mean()
    tv_f = (wf * (mask_mag[:, :, 1:] - mask_mag[:, :, :-1]).abs()).mean()
    return tv_t + tv_f
