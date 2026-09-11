# 流式语音识别输出退化分层检查设计

* **版本**：v2.0
* **日期**：2026-09-18
* **状态**：已合入主干（对应配置：`config/config.default.yaml` 中 `voice.*` 段）

---

## 1. 背景与原因分析

在长时间流式会话中，偶现前缀膨胀型的“退化行”（Degenerate Rows），表现为持续重复输出增长的前缀快照（例如 `你lo你你lo...`）。

### 原因链路分析
1. **静音误触发**：流式 Paraformer（特别是 int8 量化模型）在数字近零静音或微弱底噪环境下，声学解码器偶发吐出零星孤立 Token；
2. **时间轴漂移**：此时由于能量不足，前端 FSMN-VAD 未触发 `Speech Start`（未进入语音态），导致这些流式 Partial 的 `start_ms` 随着每次推流而不断向后漂移，且无法触发正常的断句终结；
3. **客户端累加**：部分客户端消费者（基于时间戳增量对齐逻辑）将每一次漂移的 Partial 误识别为全新的一句话，持续回放前缀快照并累加，造成字符量级膨胀；
4. **内存开销**：未决文本在会话内存中持续积累，增加不必要的状态维护开销。

---

## 2. 分层检查机制 (L0 ~ L2)

为了抑制此类退化输出，同时确保弱人声和句首辅音不被遗漏，系统设计了 L0 / L1 / L2 多层检查机制：

```
音频 Chunk 输入 (16kHz PCM16)
       │
       ▼
 [ L0: 能量检测特判 ] ──── (非语音态 且 RMS < -80 dBFS) ──► 丢弃流式 Token (抑制数字静音)
       │
       ▼
 [ L1: VAD 状态门控 ] ──── (未触发 VAD Speech Start) ────► 抑制发送 Partial (锁存首帧时间戳，累积保留)
       │
       ▼
 [ L2: 非语音超时看门狗 ] ── (未起段累积达到 8 字) ──────► 强切单段 [watermark, now] 送二遍 Qwen+Guard 裁决
       │                                                      (每会话每小时上限 20 次，封顶算力开销)
       ▼
 [ 哨兵: N-Gram 环路检测 ] ── (检测到重复循环子串) ─────────► 触发 Prometheus 告警计数
```

### 2.1 L0 检查层：近零静音能量特判
- **核心逻辑**：仅在当前处于“非语音态”（Non-speech）且当前音频 Chunk 的均方根能量（RMS）低于 `-80.0 dBFS` 时，主动丢弃流式解码器吐出的偶发 Token。代码见 [`core/audio_energy.py`](../core/audio_energy.py)。
- **阈值设定**：`-80.0 dBFS` 属于数字静音范围；模拟底噪通常在 `-70 dBFS` 到 `-60 dBFS` 之间，不影响正常环境下的微弱人声。
- **可选跳过推理**：支持 `skip_inference_on_silence: false`（默认仍执行推理，以保持流式特征提取与缓存时间线连续；配置为 `true` 可在静音时跳过推理节省计算）。

### 2.2 L1 检查层：基于 VAD 门控的 Partial 发送抑制
- **核心逻辑**：在 VAD 检测到明确的 `Speech Start` 之前，网关层抑制向客户端发送流式 Partial 消息。
- **文本暂存**：未起段期间的流式 Token 不会被丢弃，而是在内存中保持累积；一旦 VAD 随后进入语音态，首个 Partial 消息将立即携带先前累积的文本一并发出，避免漏字。
- **时间轴锁存**：对挂起状态的起始时间戳执行锁存（Latched `start_ms`），禁止随每次推流动态滑动，切断下游客户端重复拼接的条件。

### 2.3 L2 检查层：非语音累积超时看门狗与二遍裁决
- **核心逻辑**：针对 VAD 漏检的微弱语音或噪声，当非语音态下的累积字符数达到门限（默认 8 字符，`watchdog_min_chars: 8`）时，看门狗在环形缓冲区中截取 `[watermark, now]` 音频段，送入二遍 Qwen3-ASR + Consistency Guard 统一裁决。
- **频次限制**：单会话配置每小时最高允许 20 次强切（`watchdog_rate_limit_per_hour: 20`），防止异常音频频繁触发二遍推理。
- **裁决逻辑**：若 Qwen3-ASR 与一致性守卫判断其确为有效语音，则正常输出 Final 结果；若为噪声，则过滤并清空累积池。

### 2.4 输出检查：N-Gram 环路检测
- 在 Partial、Provisional 与 Final 出口处挂载滑动窗口 N-Gram 重复检测器；
- 命中重复死循环时，上报 Prometheus 监控指标 `asr_degenerate_ngram_detected_total`。

---

## 3. 配置项说明

各检查机制可通过 `config/config.default.yaml` 或环境变量进行调整：

| 配置项 | 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `voice.silence_floor_dbfs` | `VOICE_SILENCE_FLOOR_DBFS` | `-80.0` | L0 静音阈值 (dBFS)；设为 `-120.0` 可关闭此项过滤 |
| `voice.skip_inference_on_silence` | `VOICE_SKIP_INFERENCE_ON_SILENCE` | `false` | L0 静音跳过推理开关；默认保持推理以维持状态连续 |
| `voice.suppress_prestart_partials` | `VOICE_SUPPRESS_PRESTART_PARTIALS` | `true` | L1 门控开关；设为 `false` 则非语音态也发送 Partial |
| `voice.watchdog_enable` | `VOICE_WATCHDOG_ENABLE` | `true` | L2 看门狗开关 |
| `voice.watchdog_min_chars` | `VOICE_WATCHDOG_MIN_CHARS` | `8` | L2 触发字数阈值 |
| `voice.watchdog_max_segment_ms` | `VOICE_WATCHDOG_MAX_SEGMENT_MS` | `30000` | L2 单次强切最大音频长度 (毫秒) |
| `voice.watchdog_rate_limit_per_hour`| `VOICE_WATCHDOG_RATE_LIMIT_PER_HOUR`| `20` | L2 每小时强切调用次数上限 |

---

## 4. 单元测试

对应的单元测试位于 `tests/test_degenerate_guard.py` 与 `tests/test_audio_energy.py`，包括：
1. `test_l0_silence_token_drop`：近零静音下 Token 丢弃与时间线维持；
2. `test_l1_suppress_and_carry`：非语音态抑制与语音态首包完整补偿；
3. `test_l2_watchdog_force_cut`：看门狗触发、漏检语音恢复与频次限制；
4. `test_ngram_loop_canary`：重复循环模式检测与指标记录。
