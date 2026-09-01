"""多延迟分块频域自适应滤波器（MDF/PBFDAF）。"""

import numpy as np


class MDFCanceller:
    """多延迟分块频域自适应滤波器。

    调用约定：
        ref_block : 远端参考信号（喇叭播放的信号），长度 = block_size
        mic_block : 近端麦克风信号（含回声 + 近端语音 + 噪声），长度 = block_size
        返回 (err_block, echo_block)
            err_block  —— 误差信号，即“线性消回声后的近端估计”
            echo_block —— 估计出的回声
    """

    def __init__(self, block_size=128, num_partitions=32, mu=0.35,
                 forget=0.9, eps=1e-6, reg=1e-2, constrained=True,
                 far_active_rel=1e-4, leak=0.99999,
                 max_echo_mic_ratio=4.0, max_update_norm=2.0,
                 max_weight_norm=64.0):
        self.B = int(block_size)
        self.K = int(num_partitions)
        self.N = 2 * self.B                 # FFT 长度
        self.F = self.B + 1                 # rfft 频点数
        self.mu = float(mu)
        self.forget = float(forget)
        self.eps = float(eps)               # 绝对正则下限
        self.reg = float(reg)               # 相对正则项，限制低能量频点更新幅度
        self.constrained = bool(constrained)
        self.far_active_rel = float(far_active_rel)
        self.leak = float(leak)
        self.max_echo_mic_ratio = float(max_echo_mic_ratio)
        self.max_update_norm = float(max_update_norm)
        self.max_weight_norm = float(max_weight_norm)
        self._warmed = False                # 功率估计是否已热启动

        # 远端频域历史缓存 (K, F)，索引 0 为最新块
        self.X = np.zeros((self.K, self.F), dtype=np.complex128)
        # 分区滤波器权重 (K, F)
        self.W = np.zeros((self.K, self.F), dtype=np.complex128)
        # 每个频点的功率估计 (F,)
        self.P = np.zeros(self.F, dtype=np.float64)
        # overlap-save 需要保留上一块远端时域
        self.ref_prev = np.zeros(self.B, dtype=np.float64)
        self.ref_power_max = self.eps
        self.guard_activations = 0
        self.update_clips = 0
        self.weight_projections = 0
        self._guard_hold = 0

    def reset(self):
        self.X[:] = 0
        self.W[:] = 0
        self.P[:] = 0
        self.ref_prev[:] = 0
        self._warmed = False
        self.ref_power_max = self.eps
        self.guard_activations = 0
        self.update_clips = 0
        self.weight_projections = 0
        self._guard_hold = 0

    @property
    def filter_len(self):
        return self.K * self.B

    def process_block(self, ref_block, mic_block, mu_scale=1.0):
        B, N, K = self.B, self.N, self.K
        ref_block = np.asarray(ref_block, dtype=np.float64).reshape(B)
        mic_block = np.asarray(mic_block, dtype=np.float64).reshape(B)

        # --- 1. 组 2B 远端分析块(overlap-save)并做 FFT ---
        x2b = np.concatenate([self.ref_prev, ref_block])
        self.ref_prev = ref_block.copy()
        X_cur = np.fft.rfft(x2b, n=N)

        # 历史左移，最新块放到索引 0
        self.X[1:] = self.X[:-1]
        self.X[0] = X_cur

        # 远端低能量时冻结同拍更新，避免近端信号放大滤波器权重
        ref_power = float(np.mean(ref_block ** 2))
        self.ref_power_max = max(self.ref_power_max * 0.9999, ref_power, self.eps)
        far_active = ref_power > self.far_active_rel * self.ref_power_max

        # 极轻泄漏抑制长时间无可辨识更新时的权重漂移。
        if self.leak < 1.0:
            self.W *= self.leak

        # --- 2. 频域回声估计（分区加权求和）---
        Y = np.sum(self.W * self.X, axis=0)           # (F,)
        y_full = np.fft.irfft(Y, n=N)                 # (N,)
        echo_block = y_full[B:]                        # 线性卷积有效部分为后 B 点

        # 回声估计异常增大时缩放滤波器权重，并短暂冻结更新
        mic_rms = float(np.sqrt(np.mean(mic_block ** 2) + self.eps))
        echo_rms = float(np.sqrt(np.mean(echo_block ** 2) + self.eps))
        mic_peak = float(np.max(np.abs(mic_block), initial=0.0))
        echo_peak = float(np.max(np.abs(echo_block), initial=0.0))
        rms_scale = self.max_echo_mic_ratio * mic_rms / max(echo_rms, self.eps)
        peak_scale = (2.0 * self.max_echo_mic_ratio * (mic_peak + 1e-4) /
                      max(echo_peak, self.eps))
        guard_scale = min(1.0, rms_scale, peak_scale)
        if guard_scale < 1.0:
            self.W *= guard_scale
            echo_block *= guard_scale
            self.guard_activations += 1
            self._guard_hold = max(self._guard_hold, 8)

        # --- 3. 误差 ---
        err_block = mic_block - echo_block

        # --- 4. 频域自适应更新（约束式 NLMS）---
        e_pad = np.concatenate([np.zeros(B), err_block])   # 误差放后半段
        E = np.fft.rfft(e_pad, n=N)                         # (F,)

        # 每频点功率归一化（对所有分区求和）
        Xpow = np.sum(np.abs(self.X) ** 2, axis=0)         # (F,)
        if not self._warmed:
            # 首块以当前功率初始化，限制起始更新幅度
            self.P = Xpow.copy()
            self._warmed = True
        else:
            self.P = self.forget * self.P + (1.0 - self.forget) * Xpow
        # 正则项与信号尺度挂钩：绝对下限 + 平均功率的一个比例
        delta = self.eps + self.reg * float(np.mean(self.P))
        norm = 1.0 / (self.P + delta)                      # (F,)

        step = self.mu * float(mu_scale)
        if not far_active or self._guard_hold > 0:
            step = 0.0
        if self._guard_hold > 0:
            self._guard_hold -= 1
        grad = np.conj(self.X) * E[None, :]                # (K, F)
        dW = step * grad * norm[None, :]                   # (K, F)

        if self.constrained:
            # 把每个分区的更新时域支撑约束在前 B 点（因果、长度 B）
            dw_t = np.fft.irfft(dW, n=N, axis=1)           # (K, N)
            dw_t[:, B:] = 0.0
            dW = np.fft.rfft(dw_t, n=N, axis=1)

        # 限制单块更新范数，抑制低能量频点的瞬态更新
        update_norm = float(np.linalg.norm(dW))
        if update_norm > self.max_update_norm:
            dW *= self.max_update_norm / (update_norm + self.eps)
            self.update_clips += 1

        self.W += dW
        weight_norm = float(np.linalg.norm(self.W))
        if weight_norm > self.max_weight_norm:
            self.W *= self.max_weight_norm / (weight_norm + self.eps)
            self.weight_projections += 1
        return err_block, echo_block

    def process_stream(self, ref, mic, mu_scale_fn=None):
        """整段离线处理：ref/mic 为一维数组，长度需为 block_size 整数倍(不足自动补零)。

        mu_scale_fn: 可选回调 (block_index)->float，用于外部注入双讲控制。
        返回 (err, echo) 同长度一维数组。
        """
        ref = np.asarray(ref, dtype=np.float64).ravel()
        mic = np.asarray(mic, dtype=np.float64).ravel()
        n = max(len(ref), len(mic))
        pad = (-n) % self.B
        ref = np.pad(ref, (0, pad + max(0, n - len(ref))))[:n + pad]
        mic = np.pad(mic, (0, pad + max(0, n - len(mic))))[:n + pad]

        out_err = np.zeros_like(ref)
        out_echo = np.zeros_like(ref)
        nblocks = len(ref) // self.B
        for i in range(nblocks):
            s = i * self.B
            e = s + self.B
            ms = 1.0 if mu_scale_fn is None else float(mu_scale_fn(i))
            err_b, echo_b = self.process_block(ref[s:e], mic[s:e], mu_scale=ms)
            out_err[s:e] = err_b
            out_echo[s:e] = echo_b
        return out_err, out_echo
