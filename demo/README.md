# ASRFlow Client Demo

本目录提供了交互式客户端 Demo 与推流测试工具。

## 1. 工具与示例

* [`console_client.py`](console_client.py)：控制台交互式客户端。支持 ANSI 终端动态原地追加与回刷：
  - 首遍 Partial：当前句在底部活动行原位追加刷新（青色 ▶）；
  - 首遍成句：活动行固化为一条“待定稿”行（暗色 ~），换行继续下一句；
  - 二遍 Final：回刷替换对应“待定稿”行（绿色 + 说话人 + 时间戳），不改写更早历史；
  - 会话结束：汇总延迟与统计指标，支持打印完整对话转写（`--full`）。
* [`client_demo.py`](client_demo.py)：基于 Rich 终端样式的转写客户端示例，支持热词偏置与逆文本后处理展示。
* [`ws_client.py`](ws_client.py)：轻量级 WebSocket 流式推流验证脚本。
* [`fixtures/sample_16k.wav`](../fixtures/sample_16k.wav)：开箱即用的公开测试语音（Whisper JFK 片段，16 kHz mono；详见 `fixtures/README.md`）。

## 2. 快速上手

### 2.1 运行控制台实时回刷 Demo

```bash
# 1. 使用测试语料推流（默认 1.0x 实时流速）
python3 demo/console_client.py --uri ws://127.0.0.1:10095 --audio fixtures/sample_16k.wav

# 2. 以 2.0x 倍速推流快速测试
python3 demo/console_client.py --uri ws://127.0.0.1:10095 --audio fixtures/sample_16k.wav --rate 2.0

# 3. 携带会话级动态热词偏置
python3 demo/console_client.py \
  --uri ws://127.0.0.1:10095 \
  --audio fixtures/sample_16k.wav \
  --hotwords "实时语音识别,ASRFlow" \
  --full
```

### 2.2 运行简易推流测试

```bash
# 极简轻量推流测试
python3 demo/ws_client.py --uri ws://127.0.0.1:10095 --audio fixtures/sample_16k.wav
```
