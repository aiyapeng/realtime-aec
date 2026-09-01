"""AEC 训练、评测与推理的共享配置。"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Cfg:
    fs: int = 16000
    hop: int = 128
    n_fft: int = 384                     # STFT 缓冲时延为 16 ms

    # —— 线性前端 MDF：train == eval == deploy，全部走这两个值 ——
    mdf_partitions: int = 24
    mdf_mu: float = 0.4
    dtd_mu_floor: float = 0.02           # 双讲时更硬冻结（DNN 已接管 DT，原 0.05 会缓慢侵蚀近端）

    # —— 模型 ——
    hidden: int = 256
    layers: int = 3
    mask_mag_max: float = 3.0

    # —— 损失权重：w_near / w_echo 就是 DT echo ↔ DT degradation 的权衡旋钮 ——
    w_near: float = 3.0
    w_echo: float = 1.0
    w_evi: float = 0.5
    tv_w: float = 0.04

    # —— 数据增广范围（覆盖真实设备：更宽 SER/SNR、更强非线性、路径变化、时延抖动）——
    ser_db: Tuple[float, float] = (-8.0, 12.0)
    snr_db: Tuple[float, float] = (5.0, 45.0)
    nl_kinds: Tuple[str, ...] = ("soft", "clip", "sef", "memory")
    nl_strength: Tuple[float, float] = (0.05, 0.45)
    nl_pregain: Tuple[float, float] = (1.1, 2.5)
    p_path_change: float = 0.4           # 分段内回声路径 A→B 渐变的概率
    p_delay_jitter: float = 0.4          # 时变体延迟（模拟时钟漂移/路径长度变化）的概率
    warmup_sec: float = 1.5              # dt 先远端单讲让 MDF 收敛

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1


CFG = Cfg()
