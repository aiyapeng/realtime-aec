"""
delay.py
========
远端参考与麦克风信号之间的体延迟(bulk delay)估计与补偿。

AEC 里，麦克风信号相对远端参考存在“系统缓冲 + 声传播”的固定延迟，
自适应滤波器只能覆盖有限长度的尾巴，若不先对齐会白白浪费滤波器长度、
甚至无法收敛。用 GCC-PHAT（广义互相关-相位变换）估计延迟：相位变换加权
对混响/有色信号鲁棒，峰值位置即延迟。
"""

import numpy as np


def gcc_phat(sig, ref, fs=16000, max_tau=None, interp=4):
    """估计 sig 相对 ref 的延迟(秒)。sig(mic) 通常滞后 ref。

    返回 (tau_seconds, cc_curve)。tau>0 表示 sig 落后 ref tau 秒。
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = len(sig) + len(ref)
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    R /= (np.abs(R) + 1e-12)                       # PHAT 加权
    cc = np.fft.irfft(R, n=n * interp)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift + 1]])
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def estimate_bulk_delay(ref, mic, fs=16000, max_delay_ms=500):
    """估计体延迟(样点数, >=0)。只在合理正延迟范围内搜索。"""
    tau, _ = gcc_phat(mic, ref, fs=fs, max_tau=max_delay_ms / 1000.0)
    delay = int(round(tau * fs))
    return max(0, delay)


def align_ref(ref, mic, fs=16000, max_delay_ms=500):
    """把 ref 向后对齐到 mic（即给 ref 前面补 delay 个零使其与回声对齐）。

    返回 (ref_aligned, delay_samples)。ref_aligned 与 mic 等长。
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    mic = np.asarray(mic, dtype=np.float64).ravel()
    delay = estimate_bulk_delay(ref, mic, fs=fs, max_delay_ms=max_delay_ms)
    ref_aligned = np.concatenate([np.zeros(delay), ref])[:len(mic)]
    if len(ref_aligned) < len(mic):
        ref_aligned = np.pad(ref_aligned, (0, len(mic) - len(ref_aligned)))
    return ref_aligned, delay
