"""ERLE、SI-SDR、PESQ 和 STOI 指标实现。"""

import numpy as np


def erle_segmental(mic, err, win=8000, farend_active=None, floor=1e-10):
    """分段 ERLE(dB)。若给出 farend_active(布尔逐样点/逐段)，只统计远端有声段。"""
    mic = np.asarray(mic, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    n = min(len(mic), len(err))
    mic, err = mic[:n], err[:n]
    vals = []
    for s in range(0, n - win + 1, win):
        d = mic[s:s + win]
        e = err[s:s + win]
        if farend_active is not None:
            act = np.asarray(farend_active).ravel()[s:s + win]
            if np.mean(act) < 0.5:      # 该段远端基本无声, 跳过
                continue
        pd = np.sum(d ** 2)
        pe = np.sum(e ** 2)
        if pd < floor:
            continue
        vals.append(10 * np.log10(pd / (pe + floor) + floor))
    return np.array(vals)


def erle_overall(mic, err, floor=1e-10):
    mic = np.asarray(mic, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    n = min(len(mic), len(err))
    pd = np.sum(mic[:n] ** 2)
    pe = np.sum(err[:n] ** 2)
    return 10 * np.log10(pd / (pe + floor) + floor)


def si_sdr(est, ref, eps=1e-8):
    """尺度不变 SDR(dB)。est/ref 一维等长(自动截断)。"""
    est = np.asarray(est, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))


def compensate_delay(sig, delay):
    """把输出前移 delay 个样点以对齐参考(尾部补零)。
    AEC 流水线有算法时延(见 AECPipeline.latency_samples)，
    计算 SI-SDR/PESQ/STOI 这类**逐样点对齐**的指标前必须先补偿，
    否则错位会让分数假性暴跌(ERLE 用长窗能量比则不受影响)。"""
    sig = np.asarray(sig, dtype=np.float64).ravel()
    d = int(delay)
    if d <= 0:
        return sig
    return np.concatenate([sig[d:], np.zeros(d)])


def best_delay_xcorr(est, ref, max_lag=1024):
    """通过互相关估计 est 相对 ref 的整数时延。"""
    est = np.asarray(est, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    c = np.correlate(est, ref, mode="full")
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= max_lag
    return int(lags[m][np.argmax(c[m])])


def si_sdr_aligned(est, ref, delay=None, max_lag=1024):
    """时延对齐后的 SI-SDR。delay 已知则直接补偿; 否则用互相关自动估计。"""
    if delay is None:
        delay = max(0, best_delay_xcorr(est, ref, max_lag))
    return si_sdr(compensate_delay(est, delay), ref)


def try_pesq(ref, deg, fs=16000):
    """需要 `pip install pesq`。返回宽带 PESQ 或 None。"""
    try:
        from pesq import pesq
        return float(pesq(fs, np.asarray(ref), np.asarray(deg), "wb"))
    except Exception as e:  # noqa
        return None


def try_stoi(ref, deg, fs=16000):
    """需要 `pip install pystoi`。返回 STOI 或 None。"""
    try:
        from pystoi import stoi
        return float(stoi(np.asarray(ref), np.asarray(deg), fs, extended=False))
    except Exception:
        return None
