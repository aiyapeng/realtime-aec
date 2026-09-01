"""
dtd.py
======
双讲检测（Double-Talk Detection, DTD）—— 残余比 + 自适应地板法。

为什么不用“麦克风 vs 回声估计相关性”那种简单判据：滤波器冷启动时回声估计≈0，
相关性≈0 会被误判成双讲 → 冻结自适应 → 永远收敛不了（经典 DTD 冷启动陷阱）。

本实现思路（工程上鲁棒、且与滤波器收敛程度自洽）：
  * 统计远端单讲时“已达到的抵消地板”ratio_floor = min(Pe/Pd)（Pe:误差功率, Pd:麦克风功率）;
  * 只有当滤波器已收敛(地板足够低)且当前 Pe/Pd 相对地板“突然变差”超过 margin,
    才判为双讲(近端语音混入使误差变大) → 软冻结步长;
  * 远端活动门控：远端基本无声时无需(也无法)自适应, 不更新地板、不判双讲;
  * 地板允许极缓慢上抬, 使回声路径突变(看起来也像双讲)后能自动解冻、重新收敛。
输出 mu_scale∈[mu_floor,1]，直接乘到自适应步长上。
"""

import numpy as np


class DoubleTalkDetector:
    def __init__(self, forget=0.9, dt_margin_db=6.0, converged_db=6.0,
                 mu_floor=0.05, floor_rise=1.002, hold_blocks=15,
                 far_active_rel=0.01):
        self.a = float(forget)
        self.dt_margin = 10 ** (dt_margin_db / 10.0)     # 抵消变差多少倍算双讲
        self.converged = 10 ** (-converged_db / 10.0)    # 地板低于此才认为已收敛
        self.mu_floor = float(mu_floor)
        self.floor_rise = float(floor_rise)
        self.hold_blocks = int(hold_blocks)
        self.far_active_rel = float(far_active_rel)

        self.Pd = 1e-6
        self.Pe = 1e-6
        self.Px = 1e-6
        self.Px_max = 1e-6
        self.ratio_floor = 1.0
        self._hold = 0

    def reset(self):
        self.Pd = 1e-6; self.Pe = 1e-6; self.Px = 1e-6
        self.Px_max = 1e-6; self.ratio_floor = 1.0; self._hold = 0

    def update(self, mic_block, echo_block, err_block, ref_block):
        a = self.a
        Pd = float(np.mean(mic_block ** 2))
        Pe = float(np.mean(err_block ** 2))
        Px = float(np.mean(ref_block ** 2))
        self.Pd = a * self.Pd + (1 - a) * Pd
        self.Pe = a * self.Pe + (1 - a) * Pe
        self.Px = a * self.Px + (1 - a) * Px
        self.Px_max = max(self.Px_max * 0.9999, self.Px)

        far_active = self.Px > self.far_active_rel * self.Px_max

        # 远端无声：必须明确冻结。旧实现返回 1.0，导致“几乎无远端、
        # 只有近端”的块仍以满步长更新，是有限值灾难性发散的根因。
        if not far_active:
            if self._hold > 0:
                self._hold -= 1
            return 0.0, False, self.Pe / (self.Pd + 1e-9)

        ratio = self.Pe / (self.Pd + 1e-9)

        # 更新“已达到的抵消地板”
        if ratio < self.ratio_floor:
            self.ratio_floor = ratio
        else:
            self.ratio_floor = min(1.0, self.ratio_floor * self.floor_rise)

        converged = self.ratio_floor < self.converged
        if converged and ratio > self.ratio_floor * self.dt_margin:
            self._hold = self.hold_blocks         # 抵消突然变差 -> 近端出现
        elif self._hold > 0:
            self._hold -= 1

        is_dt = self._hold > 0
        mu_scale = self.mu_floor if is_dt else 1.0
        return mu_scale, is_dt, ratio
