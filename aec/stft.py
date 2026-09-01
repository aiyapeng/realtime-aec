"""
stft.py
=======
STFT/ISTFT 工具。提供两套：
  * 批处理版(numpy)：训练时对整段做变换；
  * 流式版：逐帧 push / pull，内部维护重叠相加(overlap-add)缓存，
    用于实时推理，保证与批处理数值一致。

约定：n_fft=512, hop=128, 16kHz。窗为 Hann 的平方根形式(sqrt-Hann)，
分析/合成同窗，满足 COLA(常数重叠相加)条件，重建无损。
"""

import numpy as np


def sqrt_hann(win_len):
    w = np.hanning(win_len + 1)[:-1]     # 周期化 Hann
    return np.sqrt(w).astype(np.float64)


class STFTProcessor:
    def __init__(self, n_fft=512, hop=128):
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = sqrt_hann(n_fft)
        self.F = n_fft // 2 + 1
        # sqrt-Hann + hop=n_fft/4 时，合成需再乘窗并做 COLA 归一
        # 计算 COLA 归一因子
        denom = np.zeros(n_fft)
        n_overlap = n_fft // hop
        for k in range(n_overlap):
            denom[:] += np.roll(self.win ** 2, k * hop)
        self._cola = np.median(denom[denom > 1e-8])

    # ---------- 批处理 ----------
    def stft(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        pad = self.n_fft
        xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
        nframes = 1 + (len(xp) - self.n_fft) // self.hop
        frames = np.stack([xp[i * self.hop:i * self.hop + self.n_fft] * self.win
                           for i in range(nframes)], axis=0)
        return np.fft.rfft(frames, n=self.n_fft, axis=1)   # (T, F)

    def istft(self, S, length=None):
        S = np.asarray(S)
        frames = np.fft.irfft(S, n=self.n_fft, axis=1) * self.win
        nframes = frames.shape[0]
        out_len = (nframes - 1) * self.hop + self.n_fft
        out = np.zeros(out_len)
        for i in range(nframes):
            out[i * self.hop:i * self.hop + self.n_fft] += frames[i]
        out /= self._cola
        out = out[self.n_fft:]                              # 去掉前面补的零
        if length is not None:
            out = out[:length]
        return out


class StreamingSTFT:
    """逐帧流式 STFT：每次 push 一个 hop(128点)，凑满一窗即输出一帧频谱。"""
    def __init__(self, n_fft=512, hop=128):
        self.n_fft = n_fft
        self.hop = hop
        self.win = sqrt_hann(n_fft)
        self.buf = np.zeros(n_fft, dtype=np.float64)

    def push(self, hop_block):
        hop_block = np.asarray(hop_block, dtype=np.float64).ravel()
        assert len(hop_block) == self.hop
        self.buf = np.concatenate([self.buf[self.hop:], hop_block])
        frame = self.buf * self.win
        return np.fft.rfft(frame, n=self.n_fft)            # (F,)


class StreamingISTFT:
    """逐帧流式 ISTFT：每帧频谱进来，重叠相加后吐出 hop(128点)。"""
    def __init__(self, n_fft=512, hop=128, cola=None):
        self.n_fft = n_fft
        self.hop = hop
        self.win = sqrt_hann(n_fft)
        self.ola = np.zeros(n_fft, dtype=np.float64)
        if cola is None:
            denom = np.zeros(n_fft)
            for k in range(n_fft // hop):
                denom += np.roll(self.win ** 2, k * hop)
            cola = np.median(denom[denom > 1e-8])
        self.cola = cola

    def pull(self, spec):
        frame = np.fft.irfft(spec, n=self.n_fft) * self.win
        self.ola = self.ola + frame
        out = self.ola[:self.hop] / self.cola
        self.ola = np.concatenate([self.ola[self.hop:], np.zeros(self.hop)])
        return out
