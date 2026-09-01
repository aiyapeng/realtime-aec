"""远端单讲、近端单讲和双讲场景的数据生成接口。"""

import glob
import io
import os
import time

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from .synth import loudspeaker_nl, random_rir

FS = 16000


def list_speech(root):
    fs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    return [f for f in fs if os.path.getsize(f) > 2048]   # 过滤占位/空文件


def speaker_of(path):
    """MS-SNSD/VCTK 文件名如 p234_001.wav -> 说话人 'p234'。"""
    return os.path.basename(path).split("_")[0]


def split_by_speaker(files, holdout_n, seed=0):
    """把文件按说话人划分为 (训练文件, 评测文件, 留出说话人列表)。
    留出 holdout_n 个说话人**只用于评测**, 实现严格说话人隔离。"""
    spk2files = {}
    for f in files:
        spk2files.setdefault(speaker_of(f), []).append(f)
    spks = sorted(spk2files)
    rng = np.random.default_rng(seed)
    holdout = sorted(rng.choice(spks, size=min(holdout_n, max(0, len(spks) - 1)),
                                replace=False)) if holdout_n > 0 else []
    train_files = [f for s in spks if s not in holdout for f in spk2files[s]]
    eval_files = [f for s in holdout for f in spk2files[s]]
    return train_files, eval_files, holdout


def load_wav_preserve(path, fs=FS):
    """Load audio deterministically while preserving the source amplitude.

    This is the correct loader for paired AEC-Challenge loopback/microphone/
    target signals.  Independent peak normalization changes SER/SNR and breaks
    the physical relationship between paired channels.

    The legacy :func:`load_wav` below intentionally keeps its normalization for
    single-source training augmentation, where each source is remixed later.
    """
    last_error = None
    for attempt in range(3):
        try:
            with open(path, "rb") as f:
                payload = f.read()
            x, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
            break
        except (OSError, sf.LibsndfileError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    else:
        raise RuntimeError("failed to read audio after 3 attempts: %s" % path) from last_error
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != fs:
        from math import gcd
        g = gcd(int(sr), int(fs))
        x = resample_poly(x, fs // g, int(sr) // g)
    return x.astype(np.float64)


def load_wav(path, fs=FS):
    """Fast deterministic normalized loader for independent training sources.

    Avoid importing librosa/numba inside every spawned worker; that path caused
    tens-of-seconds startup stalls on Windows and starved the GPU.
    """
    x = load_wav_preserve(path, fs=fs)
    m = np.max(np.abs(x)) + 1e-9
    return x / m * 0.7                                     # 统一到合理幅度


def make_rir_pool(n, fs=FS, seed=0, use_pra=True):
    """生成 n 条 RIR。优先 pyroomacoustics(真实几何/吸声), 失败回退合成。"""
    rng = np.random.default_rng(seed)
    pool = []
    if use_pra:
        try:
            import pyroomacoustics as pra
            for _ in range(n):
                L = [rng.uniform(3, 8), rng.uniform(3, 6), rng.uniform(2.5, 3.5)]
                rt60 = rng.uniform(0.15, 0.6)
                try:
                    e_abs, mo = pra.inverse_sabine(rt60, L)
                    room = pra.ShoeBox(L, fs=fs, materials=pra.Material(e_abs),
                                       max_order=min(mo, 12))
                    sx = [rng.uniform(0.5, L[0] - 0.5), rng.uniform(0.5, L[1] - 0.5), 1.5]
                    mx = [rng.uniform(0.5, L[0] - 0.5), rng.uniform(0.5, L[1] - 0.5), 1.2]
                    room.add_source(sx); room.add_microphone(mx)
                    room.compute_rir()
                    rir = room.rir[0][0].astype(np.float64)
                    p = int(np.argmax(np.abs(rir)))
                    rir = rir[max(0, p - 8):]              # 去掉纯传播前导零
                    rir = rir[:int(0.6 * fs)]
                    rir /= (np.max(np.abs(rir)) + 1e-9)
                    pool.append(rir)
                except Exception:
                    pool.append(random_rir(fs=fs, rt60=rt60, rng=rng))
            return pool
        except Exception:
            pass
    for _ in range(n):
        pool.append(random_rir(fs=fs, rt60=rng.uniform(0.15, 0.6), rng=rng))
    return pool


def _fit(x, n):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def build_mixture(near, far, rir, fs=FS, scenario="dt", ser_db=0.0,
                  snr_db=25.0, nl_kind="soft", warmup_sec=1.5, rng=None):
    """按场景合成一条样本，返回 dict(ref,lpb,mic,near,echo,score_start)。
      * st : 远端讲、近端静音, 全程有回声(前段自然作收敛预热)。评分取后半。
      * nst: 近端讲、**远端静音**(lpb=0 -> 无回声, 滤波器无参考不适应, 近端不被扭曲)。评分取整段。
      * dt : **先 warmup_sec 远端单讲**(有回声, 让滤波器收敛), 再进入双讲。评分只取双讲段。
    score_start: 评分应从该样点开始(dt=warmup 结束处; nst=0; st=None 表示用后半)。"""
    rng = rng or np.random.default_rng()
    n = max(len(near), len(far))
    near = _fit(near, n).copy()
    far = _fit(far, n).copy()
    w = int(warmup_sec * fs) if scenario == "dt" else 0

    if scenario == "nst":
        far_all = np.zeros(n)                       # 远端静音: 无回声、无参考
        near_all = near.copy()
    elif scenario == "st":
        far_all = far.copy()
        near_all = np.zeros(n)
    else:                                           # dt: 远端单讲预热 + 双讲
        far_warm = np.resize(far, w) if w > 0 else np.zeros(0)
        far_all = np.concatenate([far_warm, far])
        near_all = np.concatenate([np.zeros(w), near])

    # 回声 = RIR ∗ 喇叭非线性(远端); nst 远端为零 -> 回声自然为零
    ls = loudspeaker_nl(far_all, kind=nl_kind,
                        pre_gain=rng.uniform(1.1, 2.0),
                        strength=rng.uniform(0.05, 0.30))
    echo_all = fftconvolve(ls, rir)[:len(far_all)]

    # SER 缩放(仅双讲段, 相对该段回声)
    if scenario == "dt":
        reg = slice(w, w + len(near))
        pe = np.mean(echo_all[reg] ** 2) + 1e-12
        pn = np.mean(near_all[reg] ** 2) + 1e-12
        near_all = near_all * np.sqrt(pe * 10 ** (ser_db / 10.0) / pn)

    mic = near_all + echo_all

    pmic = np.mean(mic ** 2) + 1e-12
    noise = rng.standard_normal(len(mic))
    noise *= np.sqrt(pmic / (np.mean(noise ** 2) + 1e-12) * 10 ** (-snr_db / 10.0))
    mic = mic + noise

    score_start = w if scenario == "dt" else (0 if scenario == "nst" else -1)
    return dict(ref=far_all, lpb=far_all, mic=mic, near=near_all,
                echo=echo_all, score_start=score_start)


def make_example(near_path, far_path, rir, seg_len, rng, canceller_factory,
                 warmup_sec=1.0):
    """经线性 AEC 生成训练残差，返回 (err, ref, near_target) float32。
    物理场景(与评测一致): nst 远端静音; dt 先远端单讲预热再双讲。
    dt 会截取双讲段(warmup 之后)对齐, 使 DNN 学到"滤波器已收敛后的残余"。"""
    scenario = rng.choice(["st", "dt", "dt", "nst", "nst"])
    near = load_wav(near_path)[:seg_len]
    far = load_wav(far_path)[:seg_len]
    d = build_mixture(near, far, rir, scenario=scenario,
                      ser_db=rng.uniform(-3, 10), snr_db=rng.uniform(12, 35),
                      nl_kind=rng.choice(["soft", "clip"]),
                      warmup_sec=warmup_sec, rng=rng)
    aec = canceller_factory()
    err, yhat = aec.process_stream(d["ref"], d["mic"])   # yhat=线性回声估计(解耦特征)
    ref, near_t = d["ref"], d["near"]
    if scenario == "dt":                          # 只保留双讲段(滤波器已收敛)
        s = d["score_start"]
        err, yhat, ref, near_t = err[s:], yhat[s:], ref[s:], near_t[s:]
    return (_fit(err, seg_len).astype("float32"),
            _fit(ref, seg_len).astype("float32"),
            _fit(yhat, seg_len).astype("float32"),
            _fit(near_t, seg_len).astype("float32"))
