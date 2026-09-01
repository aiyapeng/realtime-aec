"""包含扬声器非线性、房间响应和背景噪声的 AEC 场景合成。"""

import numpy as np
from scipy.signal import butter, lfilter


# ----------------------- 喇叭非线性 -----------------------

def loudspeaker_nl(x, kind="soft", pre_gain=1.4, clip=0.95, strength=0.15):
    """记忆无关扬声器非线性，输入幅度按 [-1, 1] 设计。"""
    x = np.asarray(x, dtype=np.float64)
    if kind == "none":
        return x
    y = pre_gain * x
    if kind == "clip":
        return np.clip(y, -clip, clip)
    if kind == "soft":
        # 软饱和多项式(奇对称) + 硬削波, strength 控制三次项失真强度
        y = y - strength * (y ** 3)
        return np.clip(y, -clip, clip)
    if kind == "sef":
        # scaled error function 型饱和(常见于 AEC 文献)
        eta = 1.0 / max(1e-3, strength)
        return np.sign(y) * (1.0 - np.exp(-np.abs(y) * eta)) / eta
    raise ValueError(kind)


# ----------------------- 随机 RIR -----------------------

def random_rir(fs=16000, rt60=0.3, direct_gain=1.0, n_early=8, rng=None):
    """生成轻量随机 RIR：直达峰 + 稀疏早反射 + 指数衰减带限噪声尾。"""
    rng = rng or np.random.default_rng()
    length = int(max(0.05, rt60) * fs)
    h = np.zeros(length)
    # 直达
    d0 = rng.integers(20, 120)         # 直达前的体延迟(样点)
    if d0 < length:
        h[d0] = direct_gain
    # 早期反射
    for _ in range(n_early):
        t = d0 + rng.integers(30, int(0.03 * fs) + 30)
        if t < length:
            h[t] += rng.uniform(-0.6, 0.6) * np.exp(-(t - d0) / (0.01 * fs))
    # 混响尾：指数衰减 * 高斯噪声, 带限
    tail = rng.standard_normal(length) * np.exp(-np.arange(length) / (rt60 * fs / 6.9))
    b, a = butter(2, [150 / (fs / 2), 6000 / (fs / 2)], btype="band")
    tail = lfilter(b, a, tail)
    tail[:d0] = 0
    h = h + 0.3 * tail
    h /= (np.max(np.abs(h)) + 1e-9)
    return h


# ----------------------- 近端语音代用信号(无数据时的自测用) -----------------------

def synth_speech_like(n, fs=16000, rng=None, n_formants=3, syllable_rate=4.0):
    """合成带共振峰和音节包络的类语音信号，用于流水线自测。"""
    rng = rng or np.random.default_rng()
    exc = rng.standard_normal(n)
    sig = np.zeros(n)
    for _ in range(n_formants):
        f0 = rng.uniform(300, 3000)
        bw = rng.uniform(80, 250)
        lo = max(80, f0 - bw) / (fs / 2)
        hi = min(fs / 2 - 1, f0 + bw) / (fs / 2)
        b, a = butter(2, [lo, hi], btype="band")
        sig += lfilter(b, a, exc)
    # 音节级幅度包络(半整流正弦 + 随机停顿)
    t = np.arange(n) / fs
    env = np.abs(np.sin(2 * np.pi * syllable_rate * t + rng.uniform(0, 6.28)))
    gate = (rng.standard_normal(n) > -0.3).astype(float)  # 随机停顿
    # 平滑 gate
    b, a = butter(2, 20 / (fs / 2))
    gate = lfilter(b, a, gate)
    sig = sig * env * gate
    sig /= (np.max(np.abs(sig)) + 1e-9)
    return sig


# ----------------------- 混合 -----------------------

def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_at_ratio(target, interferer, ratio_db):
    """把 interferer 缩放到相对 target 的指定 dB 比(target/interferer = ratio_db)。"""
    g = _rms(target) / (_rms(interferer) + 1e-12) / (10 ** (ratio_db / 20))
    return interferer * g


def simulate_mixture(near, far, fs=16000, rir=None, nl_kind="soft",
                     ser_db=0.0, snr_db=20.0, noise=None, rng=None,
                     doubletalk_ratio=0.5):
    """合成一条 AEC 混合数据。

    参数：
        near : 近端干净语音(可为 None 表示纯单讲无近端)
        far  : 远端参考语音
        ser_db : 信回比(近端语音 / 回声)，越低回声越强；单讲可设很低(如 -20 表示近端≈0)
        snr_db : 近端语音相对噪声的信噪比
        doubletalk_ratio : 近端语音只在这一比例的时间里出现(模拟单讲/双讲交替)
    返回 dict(ref, mic, near, echo, rir, delay)
    """
    rng = rng or np.random.default_rng()
    far = np.asarray(far, dtype=np.float64).ravel()
    far /= (np.max(np.abs(far)) + 1e-9)
    N = len(far)

    if rir is None:
        rir = random_rir(fs=fs, rt60=rng.uniform(0.15, 0.5), rng=rng)

    # 远端 -> 非线性 -> 卷积 RIR = 回声
    far_nl = loudspeaker_nl(far, kind=nl_kind,
                            pre_gain=rng.uniform(1.1, 2.0),
                            strength=rng.uniform(0.05, 0.30))
    echo_full = np.convolve(far_nl, rir)[:N]

    # 近端语音：仅在部分时间出现
    if near is None:
        near_arr = np.zeros(N)
    else:
        near_arr = np.asarray(near, dtype=np.float64).ravel()
        if len(near_arr) < N:
            near_arr = np.pad(near_arr, (0, N - len(near_arr)))
        near_arr = near_arr[:N]
        near_arr /= (np.max(np.abs(near_arr)) + 1e-9)
        # 随机开一个近端活跃窗(双讲区)
        if doubletalk_ratio < 1.0:
            seg = int(N * doubletalk_ratio)
            start = rng.integers(0, max(1, N - seg))
            mask = np.zeros(N)
            mask[start:start + seg] = 1.0
            b, a = butter(2, 30 / (fs / 2))
            mask = lfilter(b, a, mask)
            near_arr = near_arr * mask

    # 按 SER 缩放回声(相对近端); 若近端全零, 直接给回声固定能量
    if np.max(np.abs(near_arr)) > 1e-6:
        echo = mix_at_ratio(near_arr, echo_full, ser_db)
    else:
        echo = echo_full * 0.5

    # 噪声
    if noise is None:
        noise = rng.standard_normal(N)
        b, a = butter(2, [100 / (fs / 2), 7000 / (fs / 2)], btype="band")
        noise = lfilter(b, a, noise)
    else:
        noise = np.asarray(noise, dtype=np.float64).ravel()
        if len(noise) < N:
            noise = np.tile(noise, int(np.ceil(N / len(noise))))
        noise = noise[:N]
    ref_level = near_arr if np.max(np.abs(near_arr)) > 1e-6 else echo
    noise = mix_at_ratio(ref_level, noise, snr_db)

    mic = near_arr + echo + noise
    peak = np.max(np.abs(mic)) + 1e-9
    if peak > 1.0:
        scale = 0.99 / peak
        mic *= scale; near_arr *= scale; echo *= scale; far *= scale

    return dict(ref=far, mic=mic, near=near_arr, echo=echo, rir=rir)
