# AEC 示例

该示例使用两段由本项目组合中的 TTS 模型生成的短语音，构造远端播放、近端说话和固定房间回声路径，不包含训练或评测数据集。

```bash
python demo/run_demo.py
```

运行后生成：

- [`far_end_reference.wav`](far_end_reference.wav?raw=1)：远端播放参考
- [`microphone_input.wav`](microphone_input.wav?raw=1)：包含回声与双讲的麦克风输入
- [`aec_output.wav`](aec_output.wav?raw=1)：完成时延补偿的 AEC 输出
- `metrics.json`：该合成示例的 ERLE、双讲 SI-SDR 与流水线时延

`metrics.json` 仅用于验证示例是否正常运行，不替代仓库中的固定评测结果。
