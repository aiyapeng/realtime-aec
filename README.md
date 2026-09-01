# Real-time Acoustic Echo Cancellation

实时 AEC 流水线由 MDF 线性前端、DTD、自适应复数掩码、时频精修网络和双讲场景门控组成。推理端以 128 点音频块运行，使用 4 帧有界前瞻和逐层卷积缓存。

## 目录

- `aec/`：MDF、DTD、时延对齐、流式 STFT/ISTFT 与端到端流水线
- `dualpath_v3.py`：因果 GRU 基础复数掩码
- `phase_aware_v6.py`：时频残差精修与场景门控
- `scripts/`：训练、流式一致性验证和实时率测试
- `checkpoints/`：基础模型、精修模型和最终实时模型
- `results/`：AECMOS、流式一致性和 CPU 基准结果

## 环境

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 快速示例

仓库提供可直接运行的合成双讲示例：

```bash
python demo/run_demo.py
```

可对比试听 [`麦克风输入`](demo/microphone_input.wav?raw=1) 与 [`AEC 输出`](demo/aec_output.wav?raw=1)。远端参考、近端参考和示例指标见 [`demo/`](demo/README.md)。

## 流式验证

```bash
python -m scripts.verify_v6_streaming --incremental-refiner
```

CPU 流水线可通过以下代码加载：

```python
from scripts.v8_cpu_runtime import create_pipeline

pipeline = create_pipeline()
print(pipeline.latency_samples)
```

基准脚本需要用户自行准备评测 bank：

```bash
python -m scripts.benchmark_v6_realtime \
  --bank-dir data/bank_test \
  --incremental-refiner
```

## 已记录结果

- 三套各 300 条未见合成样本均通过四项 AECMOS 门槛
- 最终测试：ST echo 4.5341、NST degradation 3.9307、DT echo 4.2541、DT degradation 3.8019
- i5-12600KF、PyTorch FP32 CPU：20 秒端到端 RTF 中位数 0.5695
- 固定处理时延 72 ms；分块与整段复数掩码最大误差 2.53e-7

数据集和官方 AECMOS ONNX 文件未包含在仓库中。
