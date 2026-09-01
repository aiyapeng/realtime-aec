"""MDF、DTD、神经复数掩码与流式 STFT/ISTFT 组成的实时 AEC 流水线。"""

import numpy as np

from .linear_aec import MDFCanceller
from .dtd import DoubleTalkDetector
from .stft import StreamingSTFT, StreamingISTFT


class _TorchRunner:
    """PyTorch 模型逐帧推理封装。"""
    def __init__(self, model, device="cpu", stream_chunk_frames=18,
                 incremental_refiner=False):
        import torch
        self.torch = torch
        self.model = model.eval().to(device)
        self.device = device
        self.phase_aware_stream = hasattr(model, "stream_step")
        if self.phase_aware_stream:
            self.lookahead_frames = int(getattr(model, "lookahead_frames", 0))
            self.stream_chunk_frames = int(stream_chunk_frames)
            init_lookahead = (model.init_incremental_lookahead_state
                              if incremental_refiner else model.init_lookahead_state)
            self.stream_state = (init_lookahead(batch=1, device=device)
                                 if self.lookahead_frames else
                                 model.init_stream_state(batch=1, device=device))
            self._pending = {name: [] for name in ("E", "X", "Y", "D")}
            self._mask_queue = []
            self._spec_queue = []
            self.delayed_spec = None
            self.h = None
            self.use_aux = True
            self.use_echo = True
            self.use_mic = True
        else:
            self.h = model.init_state(batch=1, device=device)
            self.use_aux = model.use_aux
            self.use_echo = getattr(model, "use_echo", False)
            self.use_mic = getattr(model, "use_mic", False)
        self.complex_mask = False

    def step(self, mag_err, mag_ref, mag_echo=None, mag_mic=None,
             spec_err=None, spec_ref=None, spec_echo=None, spec_mic=None):
        t = self.torch
        with t.no_grad():
            if self.phase_aware_stream:
                def complex_frame(value):
                    return t.from_numpy(value[None, None, :].astype("complex64")).to(self.device)
                E, X, Y, D = [complex_frame(value) for value in
                              (spec_err, spec_ref, spec_echo, spec_mic)]
                if self.lookahead_frames:
                    for name, value in zip(("E", "X", "Y", "D"), (E, X, Y, D)):
                        self._pending[name].append(value)
                    self._spec_queue.append(np.asarray(spec_err).copy())
                    if len(self._pending["E"]) == self.stream_chunk_frames:
                        values = {name: t.cat(parts, dim=1)
                                  for name, parts in self._pending.items()}
                        result = self.model.lookahead_push(
                            values["E"], values["X"], values["Y"], values["D"],
                            self.stream_state)
                        mask_re, mask_im, _, _, self.stream_state = result
                        masks = t.complex(mask_re.float(), mask_im.float())[0]
                        self._mask_queue.extend(masks.cpu().numpy())
                        self._pending = {name: [] for name in ("E", "X", "Y", "D")}
                    if self._mask_queue:
                        self.delayed_spec = self._spec_queue.pop(0)
                        return self._mask_queue.pop(0)
                    self.delayed_spec = None
                    return np.ones_like(spec_err, dtype=np.complex64)
                mask_re, mask_im, _, _, self.stream_state = self.model.stream_step(
                    E, X, Y, D, self.stream_state)
                self.complex_mask = True
                mask = t.complex(mask_re.float(), mask_im.float())
                self.delayed_spec = np.asarray(spec_err)
                return mask[0, 0].cpu().numpy()
            me = t.from_numpy(mag_err[None, None, :].astype("float32")).to(self.device)
            mr = t.from_numpy(mag_ref[None, None, :].astype("float32")).to(self.device) if self.use_aux else None
            my = t.from_numpy(mag_echo[None, None, :].astype("float32")).to(self.device) if self.use_echo else None
            mm = t.from_numpy(mag_mic[None, None, :].astype("float32")).to(self.device) if self.use_mic else None
            if self.use_mic:
                mask_re, mask_im, _, self.h = self.model(me, mr, my, mm, self.h)
                self.complex_mask = True
                mask = t.complex(mask_re.float(), mask_im.float())
            else:
                mask, self.h = self.model(me, mr, my, self.h)
        return mask[0, 0].cpu().numpy()


class _OrtRunner:
    """ONNXRuntime 逐帧推理封装(部署路径)。要求模型导出为
    输入: mag_err(1,1,F), mag_ref(1,1,F), h_in(L,1,H)
    输出: mask(1,1,F), h_out(L,1,H)。"""
    def __init__(self, onnx_path, num_layers, hidden, use_aux=True, use_echo=False):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        # 从 ONNX 图输入名自动检测所需通道(比外部传参更鲁棒)
        self.use_aux = "mag_ref" in self.in_names
        self.use_echo = "mag_echo" in self.in_names
        self.use_mic = "mag_mic" in self.in_names
        self.complex_mask = "mask_re" in [o.name for o in self.sess.get_outputs()]
        self.h = np.zeros((num_layers, 1, hidden), dtype="float32")

    def step(self, mag_err, mag_ref, mag_echo=None, mag_mic=None):
        feed = {"mag_err": mag_err[None, None, :].astype("float32"), "h_in": self.h}
        if self.use_aux:
            feed["mag_ref"] = mag_ref[None, None, :].astype("float32")
        if self.use_echo:
            feed["mag_echo"] = mag_echo[None, None, :].astype("float32")
        if self.use_mic:
            feed["mag_mic"] = mag_mic[None, None, :].astype("float32")
        outputs = self.sess.run(None, feed)
        if self.complex_mask:
            mask_re, mask_im, _, self.h = outputs
            mask = mask_re + 1j * mask_im
        else:
            mask, self.h = outputs
        return mask[0, 0]


class AECPipeline:
    def __init__(self, model=None, onnx_path=None, num_layers=2, hidden=256,
                 use_aux=True, use_echo=None, fs=16000, block=128, n_fft=384, hop=128,
                 num_partitions=24, mu=0.4, dtd_mu_floor=0.02,
                 bulk_delay=0, device="cpu", stream_chunk_frames=18,
                 incremental_refiner=False):
        assert block == hop, "本流水线取 STFT hop = 线性AEC 块长, 便于对齐"
        self.fs = fs
        self.hop = hop
        self.bulk_delay = int(bulk_delay)

        self.aec = MDFCanceller(block_size=block, num_partitions=num_partitions, mu=mu)
        self.dtd = DoubleTalkDetector(mu_floor=dtd_mu_floor)
        # use_echo: 未显式给定则从 torch 模型读取
        if use_echo is None:
            use_echo = getattr(model, "use_echo", False) if model is not None else False
        self.use_echo = use_echo
        self.stft_err = StreamingSTFT(n_fft, hop)
        self.stft_ref = StreamingSTFT(n_fft, hop)
        self.stft_echo = StreamingSTFT(n_fft, hop)
        self.istft = StreamingISTFT(n_fft, hop)

        # 远端体延迟缓冲(把 ref 延后 bulk_delay 与回声对齐)
        self._ref_buf = np.zeros(self.bulk_delay + hop, dtype=np.float64)

        self._mu_scale = 1.0    # 上一拍 DTD 决策
        # 算法时延(样点)：DNN 路径 = STFT 的 (n_fft - hop) + 体延迟；纯线性路径 = 体延迟
        self._dnn = (model is not None) or (onnx_path is not None)
        model_lookahead = int(getattr(model, "lookahead_frames", 0)) if model is not None else 0
        chunk_delay = (stream_chunk_frames - 1) if model_lookahead else 0
        self.latency_samples = self.bulk_delay + ((n_fft - hop) if self._dnn else 0) + \
            (model_lookahead + chunk_delay) * hop
        self.use_dnn = self._dnn
        if model is not None:
            self.runner = _TorchRunner(model, device=device,
                                       stream_chunk_frames=stream_chunk_frames,
                                       incremental_refiner=incremental_refiner)
        elif onnx_path is not None:
            self.runner = _OrtRunner(onnx_path, num_layers, hidden, use_aux, use_echo)
            self.use_echo = self.runner.use_echo   # 以 ONNX 图实际输入为准
        else:
            self.runner = None

    def _delay_ref(self, ref_block):
        if self.bulk_delay == 0:
            return ref_block
        self._ref_buf = np.concatenate([self._ref_buf, ref_block])
        out = self._ref_buf[:self.hop]
        self._ref_buf = self._ref_buf[self.hop:]
        return out

    def process_block(self, ref_block, mic_block):
        ref_block = np.asarray(ref_block, dtype=np.float64).ravel()
        mic_block = np.asarray(mic_block, dtype=np.float64).ravel()
        ref_al = self._delay_ref(ref_block)

        # 1) 线性 AEC (用上一拍 DTD 决定的步长)
        err_b, echo_b = self.aec.process_block(ref_al, mic_block,
                                               mu_scale=self._mu_scale)
        # 2) 更新 DTD(残余比法), 决策供下一拍使用
        self._mu_scale, _, _ = self.dtd.update(mic_block, echo_b, err_b, ref_al)

        # 3) 未配置神经后端时直接输出线性残差
        if self.runner is None:
            return err_b

        # 4) STFT -> DNN 掩蔽 -> 应用 -> ISTFT
        E = self.stft_err.push(err_b)                  # (F,) complex
        X = self.stft_ref.push(ref_al)                 # (F,) complex
        Y = self.stft_echo.push(echo_b) if self.use_echo else None  # 回声估计谱
        D = E + Y if Y is not None else E
        mask = self.runner.step(np.abs(E), np.abs(X),
                                np.abs(Y) if self.use_echo else None,
                                np.abs(D), spec_err=E, spec_ref=X,
                                spec_echo=Y, spec_mic=D)
        source_E = getattr(self.runner, "delayed_spec", E)
        S = np.zeros_like(E) if source_E is None else source_E * mask
        out_b = self.istft.pull(S)                     # (hop,)
        return out_b

    def process_stream(self, ref, mic):
        ref = np.asarray(ref, dtype=np.float64).ravel()
        mic = np.asarray(mic, dtype=np.float64).ravel()
        n = max(len(ref), len(mic))
        pad = (-n) % self.hop
        ref = np.pad(ref, (0, pad + max(0, n - len(ref))))[:n + pad]
        mic = np.pad(mic, (0, pad + max(0, n - len(mic))))[:n + pad]
        out = np.zeros_like(ref)
        for i in range(len(ref) // self.hop):
            s = i * self.hop
            out[s:s + self.hop] = self.process_block(ref[s:s + self.hop],
                                                      mic[s:s + self.hop])
        return out
