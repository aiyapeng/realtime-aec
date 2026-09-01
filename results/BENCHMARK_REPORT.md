# AEC V8 CPU 实时化最终报告

日期：2026-08-30

## 结论

V8 在永久保留 V7-final 的前提下完成 CPU 实时化。模型权重、4 帧前视、场景门控和声学输出均未改变；优化仅把时频卷积从“每 4 帧重算约 126 帧历史”改为“每层保存固定长度状态并只计算新就绪帧”。

- CPU：12th Gen Intel Core i5-12600KF
- PyTorch：2.8.0+cu126，CPU intra-op=4，inter-op=1
- 三次冷启动 20 秒端到端 RTF：0.5695、0.5717、0.5676
- RTF 中位数：0.5695；最差值：0.5717
- 相对 V7 CPU RTF 3.5952：约 6.31 倍加速
- 固定延迟：72 ms（1152 samples @ 16 kHz）
- chunk=4 流式 mask 最大误差：2.53325e-7；gate 误差：0
- chunk 1/2/3/4/5/7/16 均输出 73/73 帧且保持等价
- 三套独立 n=300 AECMOS 回归全部通过
- V8 与 V7 权重 SHA256 相同：`E3CF5024C70852922DA49E780A0D52BDD170DA815DBCDF789668F927E6A0C522`

## 指标门槛与 V8 增量路径结果

门槛：ST echo >= 4.50；NST deg >= 3.80；DT echo >= 4.00；DT deg >= 3.80。

| 数据集 | ST echo | NST deg | DT echo | DT deg | 结论 |
|---|---:|---:|---:|---:|---|
| seed120000 / n=300 | 4.5061 | 3.9076 | 4.3445 | 3.8171 | 全通过 |
| seed600000 / n=300 | 4.5024 | 3.9500 | 4.3526 | 3.9451 | 全通过 |
| seed720000 / n=300（最终盲测） | 4.5341 | 3.9307 | 4.2541 | 3.8019 | 全通过 |

与 V7 的差异只出现在两个 DT echo 的最后一位（各 +0.0001），属于卷积分块导致的浮点求和顺序差异。最薄裕量仍为 seed720000 DT deg 的 +0.0019，因此后续更换权重、量化、采样率或数据分布时仍必须重新回归。

## 优化原理

V7 的正确性实现会在每个 4 帧 chunk 上拼接历史特征，再让 7 个卷积阶段重新处理完整感受野。该方法简单可靠，但同一历史帧会被重复计算很多次。

V8 为输入卷积和每个残差块的两层卷积分别维护：

1. 该层所需的左历史与尚未满足右前视的尾部；
2. 已接收和已发射的全局帧计数；
3. 残差支路中等待同时间索引主支路输出的帧。

每次只计算新就绪帧。结束时使用与整段推理相同的零右边界冲刷尾帧。FrameGroupNorm 的统计本来就逐帧独立，因此这种拆分不会引入跨 chunk 统计差异；GRU 与场景门控仍沿用原显式状态。

## 延迟与部署参数

- STFT 固有缓冲：16 ms
- 模型前视 4 帧：32 ms
- chunk=4 调度等待：24 ms
- 合计固定延迟：72 ms

验证配置必须使用 16 kHz、hop=128、n_fft=384、chunk=4，并在新进程开始推理前设置 4 个 intra-op 线程与 1 个 inter-op 线程。推荐入口：

```python
from repro.v8_cpu_runtime import create_pipeline

pipeline = create_pipeline()
output_block = pipeline.process_block(ref_block, mic_block)  # 每块 128 samples
```

`create_pipeline()` 会加载 V8 checkpoint、设置已验证线程拓扑，并启用 `incremental_refiner=True`。若直接构造 `AECPipeline`，必须显式启用该参数，否则会回到 V7 的正确但较慢的历史重算路径。

## 关键文件

- `repro/data/phase_v8_cpu_realtime_final.pt`：V8 独立 checkpoint（权重与 V7 相同）
- `phase_aware_v6.py`：精确逐层卷积状态缓存
- `aec/runtime.py`：CPU 增量推理开关
- `repro/v8_cpu_runtime.py`：推荐 CPU 部署入口
- `repro/data/phase_v8_cpu_incremental_score_seed*.json`：三套正式结果
- `repro/data/v8_incremental_verify_chunk*.json`：多 chunk 等价性验证
- `repro/data/v8_cpu_incremental_benchmark_20s_run*.json`：三次冷启动 CPU 基准
- `V8_MANIFEST.json`：发布清单和哈希

## 边界

RTF 与 CPU、负载、电源策略、PyTorch 构建和线程竞争有关；0.5717 是本机独占测试结果，不是所有硬件的保证值。当前发布是 FP32 PyTorch CPU 路径，尚未依赖 ONNX 或量化，因此避免了量化造成的指标回归风险。AECMOS 回归证明锁定测试分布通过，不代表所有真实房间、非线性扬声器或长尾噪声均无条件通过。
