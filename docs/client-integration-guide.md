# ASRFlow 实时语音识别 · 客户端接入教程

本文面向需要接入 ASRFlow 实时转写服务的客户端开发者，完整说明 WebSocket 协议、消息格式、会话生命周期、掉线恢复（resume）、热词定制与运维接口。所有内容与当前服务端实现严格对应。

---

## 1. 服务概述（客户端视角）

ASRFlow 是**异构两遍（2-Pass）实时语音识别网关**：

| 遍 | 引擎 | 时延 | 用途 |
| :--- | :--- | :--- | :--- |
| 第一遍（流式） | Paraformer-online | 约 200~600ms | 实时字幕（`partial`）、快速成句（`provisional`） |
| 第二遍（ finalize ） | Qwen3-ASR（远程 vLLM） | 约 0.5~2s | 高精度最终文本（`final`），带标点 |

客户端只需要与 WS 网关交互：**持续推音频帧 + 接收四类识别消息**，切句、说话人分离、二遍纠错全部在服务端完成。

```
客户端                    ASRFlow 网关                     后端引擎
  │  binary PCM16 帧  ─────▶  RingBuffer + FSMN-VAD ──▶ Paraformer(流式)
  │  ◀── partial ──────────  累积增量文本                (首遍)
  │  ◀── provisional ──────  VAD 断句时 flush 首遍全句
  │                          └─ 句段音频 + 上下文 ──────▶ Qwen3-ASR(二遍)
  │  ◀── final ────────────  Guard 一致性校验后定稿
```

---

## 2. 接入信息

| 项 | 值 | 说明 |
| :--- | :--- | :--- |
| WebSocket 地址 | `ws://<host>:10095` | 音频与信令同连接 |
| HTTP 运维端口 | `http://<host>:10096` | healthz / ready / metrics / sessions |
| 服务发现 | 可选 Nacos 注册 | 多实例环境经 Nacos 发现地址，见 2.2 |
| WS 心跳 | ping 间隔 20s，超时 20s | 标准 WS 协议层 ping/pong，主流库自动应答，无需业务处理 |
| 单帧上限 | 10 MB | 超过将被连接层拒绝 |

### 2.1 音频格式要求（重要）

| 项 | 要求 |
| :--- | :--- |
| 编码 | **PCM16（有符号 16 位小端）原始字节，无 WAV 头** |
| 采样率 | **16000 Hz** |
| 声道 | **单声道** |
| 投递方式 | **Binary 帧**（Text 帧一律按信令解析，发音频文本帧会得到错误回包） |

- 采样率/声道不符会导致识别乱码或无输出，客户端必须**先重采样**（如 ffmpeg `-ar 16000 -ac 1 -sample_fmt s16`）。
- 推荐分帧：**60ms 一帧 = 1920 字节**，按实时节奏发送（每 60ms 发一帧）。服务端可承受一定倍速（实测 CPU 部署 2 倍速无积压），但超实时推流会增大延迟。
- 帧大小不要求严格 60ms（服务端内部按 600ms 攒块解码），但建议 20~200ms 之间。

换算速查：`字节数 = 毫秒数 × 32`（16000 Hz × 2 字节 ÷ 1000）。

### 2.2 通过 Nacos 服务发现接入（可选）

ASRFlow 支持在服务**完全就绪后**向 Nacos 注册临时实例（心跳保活，进程退出自动注销）。多实例部署或 IP 不固定的环境下，推荐客户端经 Nacos 动态发现服务地址，替代硬编码 `ws://<host>:10095`。

#### 服务端注册信息（客户端查询时的过滤条件）

| 项 | 默认值 | 说明（对应服务端配置） |
| :--- | :--- | :--- |
| 注册开关 | 关闭 | 服务端 `NACOS_ENABLE=true` / CLI `--nacos_enable` 开启；未开启时只能直连 |
| namespace | `public` | env `NACOS_NAMESPACE` / CLI `--nacos_namespace` |
| group | `DEFAULT_GROUP` | env `NACOS_GROUP_NAME` / CLI `--nacos_group_name` |
| serviceName | `asrflow` | env `NACOS_SERVICE_NAME` / CLI `--nacos_service_name` |
| cluster | `DEFAULT` | 仅 YAML `registry.cluster_name` 可配 |
| 实例地址 | `service_host` + **WS 端口** | 注册的端口是 **WebSocket 10095**（不是 HTTP 10096）；HTTP 运维端口未写入注册信息，按约定取 `WS 端口 + 1` 或与服务端约定 |
| 实例类型 | 临时（ephemeral） | 心跳间隔默认 5s；实例异常掉线后约 15~30s 从列表剔除 |

> 因为注册发生在 WS/HTTP 服务就绪**之后**（启动顺序保证），出现在 Nacos 列表中的实例均可接客。Nacos 的 `healthy` 仅代表心跳存活；严格场景可在建连前探 `GET http://<ip>:10096/ready` 二次确认。

#### 方式一：直接调用 Nacos Open API 查询（零依赖）

```bash
# 若 Nacos 开启了鉴权, 先登录拿 accessToken (开启鉴权时所有请求需附加 accessToken 参数):
#   curl -X POST 'http://<nacos>:8848/nacos/v1/auth/login' -d 'username=xxx&password=xxx'

curl 'http://<nacos>:8848/nacos/v1/ns/instance/list' \
  --get --data-urlencode 'serviceName=asrflow' \
  --data-urlencode 'groupName=DEFAULT_GROUP' \
  --data-urlencode 'namespaceId=public' \
  --data-urlencode 'healthyOnly=true'
```

响应（仅列关键字段）：

```json
{
  "name": "DEFAULT_GROUP@@asrflow",
  "hosts": [
    {
      "ip": "127.0.0.1",
      "port": 10095,
      "weight": 1.0,
      "healthy": true,
      "enabled": true,
      "ephemeral": true,
      "clusterName": "DEFAULT",
      "metadata": {}
    }
  ]
}
```

客户端取 `hosts[].ip + hosts[].port` 拼 `ws://{ip}:{port}` 即为 WebSocket 地址；多实例时按 `weight` 加权随机选择即可。

#### 方式二：Nacos SDK 订阅（推荐）或轮询

实际使用中建议使用 Nacos SDK（Java `nacos-client`、Go/Python 各社区版）**订阅**服务，实例上下线由 SDK 回调推送，无需轮询。无法引入 SDK 时，可用最简轮询实现：

```python
"""无 SDK 时的最简发现: 查询可用实例并随机选一个 (可每 ~10s 刷新缓存)"""
import random, requests

NACOS = "http://<nacos>:8848/nacos"

def list_ws_endpoints() -> list[str]:
    r = requests.get(f"{NACOS}/v1/ns/instance/list", timeout=3, params={
        "serviceName": "asrflow", "groupName": "DEFAULT_GROUP",
        "namespaceId": "public", "healthyOnly": True})
    r.raise_for_status()
    return [f"ws://{h['ip']}:{h['port']}" for h in r.json().get("hosts", [])
            if h.get("enabled", True)]

endpoints = list_ws_endpoints()
random.shuffle(endpoints)
uri = endpoints[0]      # -> 交给 websockets.connect(uri, ...)
```

#### 多实例下的负载均衡与重连（务必阅读）

- 会话状态（句子记录、说话人聚类、时间轴）保存在**单个 ASRFlow 实例内存中，实例间不共享**；
- 结合第 7 节 resume：**断线重连必须优先回到原实例**。客户端应缓存断线前的 `ip:port`,先对原地址做指数退避重试；
- 仅当原实例从 Nacos 列表中消失（下线）才切换到其它实例，此时会话必然过期，按**全新会话**处理（丢弃补发缓冲，见 7.2 第 4 步）;
- 新建会话（非 resume）可任意挑选列表中的健康实例，天然负载均衡。

---

## 3. 客户端 → 服务端

### 3.1 JSON 信令（推荐）

所有 JSON 信令均为 Text 帧，UTF-8 编码。

#### start —— 初始化（或恢复）会话

```json
{
  "type": "start",
  "session_id": "my-uuid-001",        // 可选，客户端自定义；缺省服务端生成
  "language": "zh",                    // 可选，默认 zh（预留字段）
  "enable_spk": true,                  // 可选，默认 true，开启说话人分离
  "expected_speakers": 2,              // 可选，默认 2，预期说话人数（抑制 SPK 漂移）
  "hotwords": ["星巴克", "电影院"],   // 可选，建会话热词，见第 8 节
  "resume": true                       // 可选，掉线重连恢复，见第 7 节
}
```

回包：`session_ready`（新建）或 `session_resumed`（恢复成功）。

#### commit —— 手动断句

```json
{"type": "commit"}
```

不等 VAD 静音，立即把当前语音段出句（provisional + final）。适合"说完一句点一下"或对讲机场景。

> 注意：当前语音段不存在（静音中）时，commit 仍会以**最近 2 秒**为窗口出句——该段通常被二遍判定为无有效语音而定稿空文本（见 4.4），客户端不应据此报错。

#### stop —— 结束会话

```json
{"type": "stop"}
```

服务端会：flush 尾部语音 → 等待所有在途二遍任务 → 说话人全局重聚类 → 下发 `session_finished`（含完整转写）→ 移除会话。

#### ping

```json
{"type": "ping"}
```

回包 `{"type":"pong","event":"pong"}`。业务层探活用（与 WS 协议层心跳无关）。

### 3.2 FunASR 风格文本指令（兼容）

以下**纯文本**指令与 JSON 信令等价，兼容 FunASR realtime 客户端：

| 指令 | 等价 JSON | 回包 |
| :--- | :--- | :--- |
| `START` | `{"type":"start"}` | `session_ready` |
| `COMMIT` | `{"type":"commit"}` | （触发出句流） |
| `STOP` | `{"type":"stop"}` | `session_finished` 后追加 `{"event":"stopped"}` |
| `PING` | `{"type":"ping"}` | `{"type":"pong","event":"pong"}` |
| `HOTWORDS:词1,词2` | — | `{"event":"hotwords_set","hotwords":[...]}` |
| `POSTPROCESS_HOTWORDS:错=>对` | — | `{"event":"postprocess_hotwords_set","hotwords":{...}}` |
| `LANGUAGE:中文` | — | `{"event":"language_set","language":"中文"}`（预留） |

### 3.3 未 start 直接推音频

Binary 帧先于任何 start 到达时，服务端会**自动创建默认会话**（enable_spk=true、expected_speakers=2、无热词）并回 `session_ready`。推荐显式 start 以便自定义参数与携带 session_id。

---

## 4. 服务端 → 客户端

### 4.1 session_ready

```json
{
  "type": "session_ready",
  "event": "started",
  "session_id": "my-uuid-001"
}
```

自动建会话路径无 `event` 字段。**收到此消息后即可推音频**（若音频先于 start 到达，服务端会自动创建默认会话并同样回 `session_ready`，见 3.3）。

### 4.2 partial —— 流式字幕（第一遍）

```json
{
  "type": "partial",
  "mode": "2pass-online",
  "session_id": "my-uuid-001",
  "seq": 42,                  // partial 递增序号
  "start_ms": 15200,          // 当前句起始（绝对时间轴，毫秒）
  "end_ms": 18600,            // 当前帧到达时间
  "begin_time": 15200,        // 同 start_ms（兼容字段）
  "end_time": 18600,          // 同 end_ms（兼容字段）
  "text": "现在是二零二五年十二月",   // 当前句累计文本（同一句内递增）
  "wav_name": "",
  "is_final": false
}
```

要点：**`text` 是当前句内的累积文本**（非增量），渲染字幕直接整体替换即可；每约 600ms 更新一次；换句后从空重新增长。

### 4.3 provisional —— 成句快照（第一遍）

VAD 检测到句尾（或 commit / 强制切分）时下发：

```json
{
  "type": "provisional",
  "mode": "2pass-provisional",
  "session_id": "my-uuid-001",
  "sentence_id": 7,           // 会话内句序号，从 1 递增
  "start_ms": 15200,
  "end_ms": 19300,
  "begin_time": 15200,
  "end_time": 19300,
  "text": "现在是二零二五年十二月十日晚上八点四十五分",
  "wav_name": "",
  "is_final": false
}
```

### 4.4 final —— 定稿结果（二遍，权威文本）

```json
{
  "type": "final",
  "mode": "2pass-offline",
  "session_id": "my-uuid-001",
  "sentence_id": 7,           // 与对应 provisional 相同
  "start_ms": 15200,
  "end_ms": 19300,
  "begin_time": 15200,
  "end_time": 19300,
  "speaker": "SPK1",          // 说话人标签
  "text": "现在是2025年12月10日晚上8点45分。",   // ITN+热词后处理后的最终文本
  "final_source": "qwen3-asr",  // "qwen3-asr" | "paraformer-fallback"
  "revision_distance": 0.18,    // 两遍编辑距离 0~1，越小越一致
  "needs_review": false,        // true 表示两遍分歧较大，建议人工复核
  "wav_name": "",
  "is_final": true
}
```

- `final_source="qwen3-asr"`：二遍成功返回（含空文本）。Consistency Guard **不会**因幻觉类异常改写 `final_source` 或换成首遍文本；异常时仍保留 Qwen 文本并置 `needs_review=true`（首遍质量显著差于二遍，换文几乎总是更差）。
- `final_source="paraformer-fallback"`：仅当二遍**明确超时 / 失败 / 不可用**（含待处理队列溢出，上限 `final_asr.max_queue_size` 默认 64）时，流水线例外路径回退首遍 provisional 文本。出现该值说明二遍链路有问题或积压，值得排查。**不是** Guard 幻觉判定的结果。
- `text=""` 且 `final_source="qwen3-asr"`：二遍成功响应但判定该段**无有效语音**（常见于极短语气词/噪声段），空文本即为权威结果，客户端应原样落稿而非回退首遍。
- `needs_review=true`：Guard 标记可疑（两遍编辑距离过大、语速超标、超短音频出长文本、重复环、异常插入等），**文本仍为 Qwen 定稿**；可按业务决定是否人工复核或后处理。

简表：

| 条件 | `final_source` | `needs_review` | `text` 来源 |
| :--- | :--- | :--- | :--- |
| 二遍成功且 Guard 通过 | `qwen3-asr` | `false` | Qwen |
| 二遍成功但 Guard 异常 | `qwen3-asr` | `true` | Qwen（保留） |
| 二遍超时/失败/队列溢出 | `paraformer-fallback` | `false` | 首遍 provisional |

### 4.5 session_finished —— 会话结束汇总

```json
{
  "type": "session_finished",
  "mode": "2pass-offline",
  "event": "stopped",
  "session_id": "my-uuid-001",
  "total_sentences": 18,
  "duration_ms": 60000,        // 会话音频总时长
  "sentences": [ ... ],        // SentenceRecord 数组（同 final 的字段集）
  "transcript": [ ... ],       // 同 sentences（兼容字段）
  "is_final": true
}
```

### 4.6 session_resumed —— 恢复成功（见第 7 节）

```json
{
  "type": "session_resumed",
  "event": "resumed",
  "session_id": "my-uuid-001",
  "last_end_ms": 19300,        // 服务端已记录句子(含断线时在途句)的最大 end_ms，客户端从此处重放音频
  "total_sentences": 7,
  "sentences": [ ... ],        // 断线时已记录的全部句子(同 final 的字段集); 在途句 is_committed=false 且 final_text 可能为空
  "is_final": false
}
```

> 在途句的 final 若在断线期间/重连后才完成，**不会重新推送**——其结果已写入会话记录，体现在后续 `session_finished` 与落盘转写中；`sentences` 中该句的 `final_text` 以 resume 时刻为准。

### 4.7 error

```json
{"type": "error", "code": 4002, "message": "Unknown command 'xxx'"}
```

| code | 含义 |
| :--- | :--- |
| 4000 | Text 帧不是合法 JSON/指令 |
| 4002 | 未知命令类型 |
| 4004 | 无会话时 commit/stop |

连接数超过服务端 `max_connections` 时，新建会话请求会使连接被服务端直接关闭（无 error 帧），客户端应按"服务过载"处理并退避重试。

---

## 5. 完整交互流程

```
 客户端                                服务端
   │                                     │
   │── Text: {"type":"start",...} ─────▶│ 创建会话
   │◀────────── session_ready ──────────│
   │                                     │
   │── Binary: PCM16 60ms ─────────────▶│ 写环形缓冲，VAD+Paraformer 并行
   │── Binary: PCM16 60ms ─────────────▶│
   │◀── partial (累计字幕) ─────────────│ 每 ~600ms 更新
   │── Binary: ... ────────────────────▶│
   │◀── partial ────────────────────────│
   │                                     │ ← VAD 检测到 ≥max_end_silence_time 静音
   │◀── provisional (首遍成句) ──────────│     或连续语音达到 max_speech_duration_ms 强制切分
   │                                     │ 句段音频异步提交二遍队列
   │◀── final (二遍定稿, SPK/距离/来源) ──│ Guard 校验后定稿
   │           ... 循环 ...               │
   │── Text: {"type":"stop"} ──────────▶│ flush 尾句 + 等待在途二遍 + 重聚类
   │◀── final (尾部句) ─────────────────│
   │◀── session_finished (完整转写) ─────│ 会话移除（如配置则落盘 JSON）
   │                                     │
```

时间预期（CPU 部署实测参考）：

- partial:语音开始后 < 1s 出现，之后每 ~600ms 刷新；
- provisional:句尾静音后 ~100ms 内；
- final:provisional 后 0.5~2s（取决于二遍服务负载）。

---

## 6. 切句行为与客户端可感知的调参

句子的粒度由服务端两个参数决定（客户端无法在协议内修改，接入前与服务端约定）：

| 参数 | 默认 | 行为 |
| :--- | :--- | :--- |
| `vad.max_end_silence_time` | 800ms | 停顿超过该值即断句。朗读/宣读类音频停顿短（300~700ms），可调小（如 480ms） |
| `vad.max_speech_duration_ms` | 60000ms | 连续无停顿语音的**单句上限**，达到即强制切分（`<=0` 禁用） |

**接入方务必确认 `max_speech_duration_ms` 不超过二遍服务的音频长度上限**。例如 Qwen3-ASR vLLM 部署实测约 28s 截断，则应设 `--max_speech_duration_ms 28000`，否则超长句的二遍文本会被截断。

---

## 7. 掉线重连与 resume

### 7.1 机制

| 阶段 | 行为 |
| :--- | :--- |
| 意外断开（未发 stop） | 会话**挂起保留**（句子记录、说话人聚类、音频时间轴），宽限期 `server.resume_ttl_sec`（默认 30s） |
| 宽限期内重连 | `{"type":"start","session_id":"...","resume":true}` → 续接同一会话，回 `session_resumed` |
| 宽限期已过 | 同样的 resume 请求**等同新建连接**：回 `session_ready`（全新会话、时间轴从 0 开始，无需重放） |
| 一直未重连 | 服务端自动收尾（完成在途二遍 + 重聚类，配置了 transcript_dir 则落盘） |
| 正常 stop 结束 | 不可 resume |

### 7.2 客户端职责：重放音频缺口

断线期间服务端收不到音频，**缺口必须由客户端补发**。标准做法：

1. 客户端维护一个**发送环形缓冲**，缓存尚未被 final 覆盖的音频（建议缓存 ≥ 会话可能断线时长）；
2. 重连成功收到 `session_resumed` 后，取 `last_end_ms`，将缓冲中 **`last_end_ms` 之后**的音频按时间顺序重放（重放可适当加速，服务端允许超实时推流）；
3. 重放期间正常接收新的 partial/final（时间轴在原会话上继续延伸）；
4. 若收到的是 `session_ready`（超时已过期），丢弃缓冲，作为全新会话开始采集。

```text
断线前:  [====已定稿====][==已发送未定稿==][未发送] ▶ 时间轴
                              last_end_ms ▲
重连后重放:                    ◀── 从这里开始补发 ──▶ 接着推实时音频
```

### 7.3 重连策略建议

- 指数退避：1s / 2s / 4s ... 上限 30s；总时长超过 resume_ttl_sec 后仍可继续尝试（自动降级为新会话）；
- 每次重连必须携带当初 start 使用的同一 `session_id` 与 `resume:true`；
- WS 库的协议层 ping/pong 保活失败即触发重连逻辑（服务端 20s 无 pong 判死）；
- **Nacos 多实例部署时，重连必须优先直连断线前的同一实例**（会话状态不跨实例共享），确认原实例已下线才换实例并以新会话开始，见 2.2 最后一节。

---

## 8. 热词与文本定制（以会话为维度）

热词**会话间完全隔离**，三种注入方式：

1. **建会话时**：`start.hotwords`；
2. **会话中随时**：Text 帧 `HOTWORDS:词1,词2`（整体替换）——对首遍 Paraformer 解码偏置与二遍 Qwen prompt **即时生效**；回包 `hotwords_set`；
3. **确定性后处理**：`POSTPROCESS_HOTWORDS:错=>对,别字=>正字`——不做解码偏置，在 final 文本上做精确替换，适合"稳定错一类"的纠正；回包 `postprocess_hotwords_set`。

建议：专有名词（人名、地名、品牌名）用 hotwords；同音固定错字用 postprocess。热词数量建议 ≤ 20（二遍 prompt 仅取前 5 个）。

---

## 9. 示例代码

### 9.1 Python 最小完整客户端

```python
"""ASRFlow 实时转写最小客户端：推 mp3/wav 并打印全部识别消息。"""
import asyncio, json, subprocess, sys
import websockets

URI = "ws://127.0.0.1:10095"

def load_pcm16_16k(path: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", "-f", "s16le", "pipe:1"],
        capture_output=True, check=True).stdout

async def main(audio_path: str):
    pcm = load_pcm16_16k(audio_path)
    chunk, interval = 1920, 0.06          # 60ms/帧，实时节奏

    async with websockets.connect(URI, max_size=10 * 1024 * 1024) as ws:
        async def receiver():
            async for raw in ws:                    # 所有回包都是 JSON Text 帧
                msg = json.loads(raw)
                t = msg.get("type", msg.get("event", ""))
                if t == "partial":
                    print(f"\r[字幕] {msg['text']}", end="", flush=True)
                elif t == "provisional":
                    print(f"\n[成句#{msg['sentence_id']}] {msg['text']}")
                elif t == "final":
                    print(f"[定稿#{msg['sentence_id']}] ({msg['speaker']}, "
                          f"{msg['final_source']}) {msg['text']}")
                elif t == "session_finished":
                    print(f"[会话结束] {msg['total_sentences']} 句 / "
                          f"{msg['duration_ms']/1000:.1f}s")

        recv_task = asyncio.create_task(receiver())
        await ws.send(json.dumps({
            "type": "start", "session_id": "demo-001",
            "enable_spk": True, "expected_speakers": 2,
            "hotwords": ["星巴克", "电影院"],
        }))
        await asyncio.sleep(0.3)                    # 等 session_ready

        for i in range(0, len(pcm), chunk):         # 按实时节奏推流
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(interval)

        await ws.send(json.dumps({"type": "stop"}))
        await recv_task

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/sample_16k.wav"))
```

### 9.2 带断线重连与补发的客户端骨架

```python
class ResumableClient:
    """缓存未定稿音频，断线后 resume 并从 last_end_ms 补发。"""

    def __init__(self, uri, session_id):
        self.uri, self.session_id = uri, session_id
        self.buf = {}                 # offset_ms -> bytes（发送环形缓冲）
        self.last_end_ms = 0

    async def run(self, mic_stream):
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.uri) as ws:
                    backoff = 1.0
                    await ws.send(json.dumps({
                        "type": "start", "session_id": self.session_id,
                        "resume": True, "hotwords": [...],
                    }))
                    await self._pump(ws, mic_stream)      # 收发主循环
            except (websockets.ConnectionClosed, OSError):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _pump(self, ws, mic_stream):
        send_task = asyncio.create_task(self._sender(ws, mic_stream))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "session_resumed":
                    # 从 last_end_ms 起补发缓存的音频，再继续实时推流
                    await self._replay_from(ws, msg["last_end_ms"])
                elif msg.get("type") == "session_ready":
                    self.buf.clear()                      # 已过期，全新会话
                elif msg.get("type") == "final":
                    self.last_end_ms = max(self.last_end_ms, msg["end_ms"])
                    self._trim_buffer()
        finally:
            send_task.cancel()

    async def _sender(self, ws, mic_stream):
        async for chunk, offset_ms in mic_stream:         # (音频, 时间戳)
            self.buf[offset_ms] = chunk                    # 先缓存再发送
            await ws.send(chunk)

    async def _replay_from(self, ws, last_end_ms):
        for offset_ms in sorted(k for k in self.buf if k >= last_end_ms):
            await ws.send(self.buf[offset_ms])

    def _trim_buffer(self):
        for k in [k for k in self.buf if k < self.last_end_ms]:
            del self.buf[k]
```

### 9.3 Node.js 简例

```js
const WebSocket = require("ws");
const ws = new WebSocket("ws://127.0.0.1:10095");

ws.on("open", () => ws.send(JSON.stringify({ type: "start", session_id: "node-1" })));
ws.on("message", (data) => {
  const msg = JSON.parse(data);              // 音频帧为 Binary，其余为 JSON 文本
  if (msg.type === "session_ready") startMic();
  if (msg.type === "partial") renderSubtitle(msg.text);
  if (msg.type === "final") appendTranscript(msg);
  if (msg.type === "session_finished") finish(msg);
});
// startMic(): 采集 16kHz/mono/PCM16，每 60ms ws.send(Buffer.from(pcm))
```

浏览器端注意：`AudioContext` 默认 48kHz/float32，需 `AudioWorklet` 降采样重编码为 PCM16 后发送。

---

## 10. HTTP 运维接口（10096 端口）

| 路径 | 说明 |
| :--- | :--- |
| `GET /` 或 `/dashboard` | 内置运维面板：含「音频文件推流」（浏览器解码重采样为 PCM16/16kHz、倍速/暂停/进度、自动 STOP）、麦克风采集与实时转写 |
| `GET /healthz` | 进程存活 |
| `GET /ready` | 模型全部加载完毕，可接客（**就绪探针用这个**） |
| `GET /metrics` | Prometheus 指标（含 `asr_paraformer_latency_ms`、`asr_sentences_finalized` 等） |
| `GET /api/v1/sessions` | 会话列表，响应 `{"total": N, "sessions": [...]}`；每项含 session_id / created_at / last_active_at / idle_seconds / sentence_count / is_in_speech / **detached**（挂起待 resume） |

---

## 11. 常见问题与排错

| 现象 | 原因与处理 |
| :--- | :--- |
| 收不到任何 partial | 音频不是 PCM16/16k/mono（发了 WAV 头或 mp3 原文）；或用 Text 帧发了音频 |
| partial 不增长 | 当前处于静音段；确认音频确实含语音且未静音丢弃 |
| 只有 provisional 没有 final | 二遍服务不可达/超时（final_source=paraformer-fallback 仍会有文本）；检查 `final_asr.vllm_url` 与远端健康 |
| final 的 text 为空 | 二遍成功响应但判定该段无有效语音（极短语气词/噪声段），空文本即权威结果，正常落稿即可 |
| 长句二遍文本被截断 | 单句超过二遍模型音频上限（Qwen3-ASR vLLM 实测 ~28s）；让服务端调小 `max_speech_duration_ms` |
| 收不到 session_finished | 忘了发 stop；或 stop 后在途二遍较多，等待 `finish_timeout`（建议 ≥ 10s） |
| resume 后时间轴跳变 | 未按 `last_end_ms` 补发，或收到的是 `session_ready`（已过期新会话）仍当作 resume 处理 |
| 连接建立后立即被关闭 | 达到 `max_connections` 上限，服务过载，退避后重试 |
| 识别文本错字固定出现 | 用 `POSTPROCESS_HOTWORDS:错=>对` 精确替换；专有名词用 hotwords |

---

## 12. 消息速查卡

```
客户端发送                                   服务端回包
─────────────────────────────              ─────────────────────────────
{"type":"start"[,session_id,...]}      →   session_ready / session_resumed
Binary: PCM16 16k mono (60ms)          →   partial(累积字幕) / provisional(成句)
                                              └→ final(定稿, 异步 0.5~2s)
{"type":"commit"}                      →   provisional + final（立即断句）
{"type":"stop"}                        →   final(尾句) + session_finished
{"type":"ping"}                        →   pong
HOTWORDS:a,b                           →   hotwords_set
POSTPROCESS_HOTWORDS:错=>对            →   postprocess_hotwords_set
非法 Text 帧                           →   error(4000/4002/4004)
断线(未 stop)                          →   会话挂起 resume_ttl_sec；重连 resume:true
                                           → session_resumed(last_end_ms) / session_ready(过期)
```
