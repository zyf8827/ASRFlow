# ASRFlow 首遍推理（Pass-1）全版本对比 Benchmark 报告

- **日期**：2026-09-04
- **范围**：仅首遍（Pass-1）流式推理实现，不含二遍（Qwen3-ASR/vLLM）与网关二遍链路
- **环境**：Intel Xeon Gold 6148 @2.40GHz，24 vCPU，62.9 GB 内存；小参数 Paraformer-online (`speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online` v2.0.4)，CPU 推理，chunk_size [5,10,5]，60ms 推流帧，16k 语音样本切片
- **代码版本**：
  - 原版 / 多实例：git `9bfd24f` (main)
  - 方案 B（torch 张量层合批，原型）：`feat/streaming-batched` @ `bca08ea`（worktree `../asrflow-batched`）
  - 方案 C（ONNX Runtime 合批，推荐）：`feat/streaming-onnx` @ `5123296`（worktree `../asrflow-onnx`）
- **原始数据**：
  - 今日三组：`benchmark/results/batch_compare/20260904_014134_paraformer_engine/`（原版，主仓）、`.../asrflow-batched/benchmark/results/batch_compare/20260904_015408...`（B）、`.../asrflow-onnx/benchmark/results/batch_compare/20260904_014635...` 与 `20260904_015615...`（C 含 2 进程组）
  - 多实例/多进程基线：引用 `docs/benchmark_report_2026-09-03.md`（git `5419fe2`）

---

## 1. 执行摘要

| 版本 | 首遍单进程吞吐上限 | 单进程实时承载 | 100 路所需(24vCPU 单机) | 正确性 | 状态 |
|---|---:|---:|---:|---|---|
| 原版（锁串行） | **8.0x**（加流不涨） | 4 路稳定 / 8 路临界掉队 | ~13 进程（超机容量）→ 多机 | 现网基线 | 生产现状 |
| 原版 + 多实例负载均衡 | 6 进程 41.8x（近线性） | 6 实例×4 路=24 路/机（实测） | ~4-5 台同规格机器 | 同上 | 9/3 报告结论 |
| 方案 B（torch 张量层合批） | **6.1x（低于基线）** | 未测（不推荐） | — | 残 ~1 字级不一致 | **研究原型，不合入** |
| **方案 C（ONNX int8 合批）** | **31.2x（16 路合批）**；2 进程 46.6x | **16 路按时完成** | **3~4 进程（单机）** | **与官方单流参考逐句一致** | **推荐生产路径** |

**一句话结论**：首遍瓶颈（推理锁串行化）的解法已从"横向堆进程"升级为"进程内多路合批"——方案 C 用官方导出 ONNX 图的动态 batch 轴，把单进程吞吐从 8x 提到 31x（3.9 倍）、单机 100 路从"需要 4~5 台机器"降到"单机 3~4 个进程"，同时模型内存从 1152MB 降到 ~15MB（int8 图 82MB）。方案 B（绕过官方 API 直接批 torch 张量）数学上可行但工程与性能双输，保留为研究记录。

---

## 2. 被测版本说明

### 2.1 原版（`ParaformerStreamingEngine`，锁串行）
每副本一把推理锁，进程内多路流串行化为同一份推理算力。9/3 报告已证明：单进程吞吐恒定 ~8x 实时（1 路 7.97x vs 4 路 8.18x），加流只增加锁等待（4 路 p95 71ms）。

### 2.2 原版 + 多实例负载均衡（9/3 报告结论，口径：threads/proc=2）
- 多进程 1/2/4/6 进程近线性：7.97 → 14.51 → 28.94 → **41.79x**（并行效率 87~91%）；12 进程塌缩至 11.32x。
- 实时节奏：6 进程×4 路=24 路按时完成（本机实际上限区）；12 进程×4 路崩溃（4 倍慢于实时）。
- 网关级（完整 WS 链路）：单实例稳定 4 路（8 路临界，7/8 成功）；4 实例×16 路全部成功。

### 2.3 方案 B：torch 张量层合批（`BatchedStreamingEngine`，`feat/streaming-batched`）
绕过官方 `inference()` 的 batch=1 断言，直接驱动 encoder/predictor/decoder 张量模块，单前向线程沿 batch 维合并多路同形请求。
- **可行性已证明**：合批与逐路的 encoder 输出**位级一致**；修复了 decoder FSMN 零填充污染（按 CIF token 数分组执行 decoder）后，双字乱码消失。
- **（更新）残留差异已澄清为伪影**：所谓 ~1 字级不一致源于一致性测试的 solo 参考一直在随机 dither 下运行——`AutoModel.generate` 内部使用独立于 `kwargs["frontend"]` 的 frontend 实例，对其 patch dither 无效。改在 `extract_fbank` 入口对实际实例置零后，引擎与官方路径**逐句完全一致**（4/6 路并发、20s 长段、无中间 flush 场景全部 PASS）。
- **性能倒退**（见 §3）：future 队列往返 + torch CPU 合批摊薄不足 + cache 合并/拆分 Python 开销，1 路 4.6x（-44%）、8 路 6.1x（-24%）。

### 2.4 方案 C：ONNX Runtime 合批（`OnnxBatchedStreamingEngine`，`feat/streaming-onnx`）
官方 funasr 导出图（`model[_quant].onnx` + `decoder[_quant].onnx`，输入签名带动态 `batch_size` 轴——官方 C++ runtime 生产上即对同一批图喂 batch>1），调用语义严格镜像官方 Python 参考 `funasr_onnx`。单模型 + 单前向线程，多路同形 chunk 沿 batch 维合批；每流独立 frontend（dither=0 确定性）；decoder 按 CIF token 数分组执行（关键正确性修复）。
- **正确性**：fp32 单路/多路并发与官方单流参考**逐句完全相等**；int8 量化图 4 路并发 CER=0.0；网关端到端冒烟（真实 FSMN-VAD + mock 二遍）partial/final 文本与直连逐字一致（`scripts/smoke_onnx_gateway.py`）。
- 与 torch 路径存在 0~9% CER 的句边界差异：两套官方实现的 is_final 策略不同（ONNX 两段切分 vs torch tail_chunk），属预期语义差异非缺陷。

---

## 3. 引擎级实测（今日三组，同机同口径：单进程、threads=8、每路 30s）

### 3.1 offline 全速吞吐（音频秒/墙钟秒）

| 每进程流数 | 原版 | 方案 B | 方案 C（int8） |
|---:|---:|---:|---:|
| 1 | 8.20x | 4.62x | 11.02x |
| 4 | 8.13x | 6.52x | **22.36x** |
| 8 | 8.04x | 6.10x | **24.32x** |
| 16 | — | — | **31.16x** |

- 原版：吞吐与流数无关（锁串行化铁顶，与 9/3 结论一致）。
- 方案 B：全档位低于原版（合批收益 < 调度开销），不具生产价值。
- 方案 C：随流数近线性抬升（合批生效），16 路 31.2x 为原版 8 路容量的 3.9 倍。

### 3.2 realtime 实时节奏承载（按真实语速推流）

| 组合 | 原版 | 方案 C（int8） |
|---|---|---|
| 1×1 | 1.00x，健康 | 1.00x，健康 |
| 1×4 | 3.93x 按时，**排队 p95 194ms** | 3.98x 按时，**排队 p95 16ms** |
| 1×8 | 5.53x，**墙钟 43s 掉队**（30s 音频） | 7.93x，**墙钟 30.3s 按时**，call p99 185ms |
| 1×16 | — | **15.69x，按时完成**，call p50 0.13ms / p99 335ms |

- 原版 8 路已进入积压区（端到端 1.4 倍拉长）；方案 C 16 路仍按时完成。
- 方案 C 的快路径（每 60ms 帧仅 1/10 触发前向）使 call p50 仅 0.13ms。

### 3.3 多进程扩展性（方案 C 补测 + 引用 9/3 原版多进程）

| 拓扑 | 原版（9/3，threads=2） | 方案 C（今日，threads=8） |
|---|---:|---:|
| 1 进程 | 7.97x | 31.16x（16 路） |
| 2 进程 | 14.51x（1.82x 加速） | **46.65x**（2×8 路） |
| 4 进程 | 28.94x（3.63x） | 未测（预计 ~80-90x） |
| 6 进程 | 41.79x（5.24x） | 未测 |

> 口径说明：9/3 原版多进程组为 threads/proc=2（其矩阵以吞吐为目标），今日 C 组为 threads/proc=8；两组各自内部自洽，跨组绝对值比较以今日同口径三组（§3.1/3.2）为准，本表仅展示"合批后多进程扩展性保持"这一结论（2 进程加速比 1.5x，与 threads 占用 16/24 vCPU 一致）。

---


### 3.4 补遗（V2 主仓池化副本实测，本日新增；详见 PDF 版报告）

用引擎工厂直测 `PooledStreamingEngine`（benchmark/bench_v2_pool.py，threads=2）：

| 进程内副本数 × 流数 | offline 吞吐 | call p99 | 结论 |
|---|---:|---:|---|
| 1 副本 × 4 路 | **8.42x** | 82ms | 与 V1 单进程等价（交叉校验 ✓） |
| 2 副本 × 4 路 | 7.17x | 217ms | 不升反降 |
| 4 副本 × 8 路 | 6.34x | 437ms | 同上 |
| 6 副本 × 12 路 | 6.64x | 647ms | 同上（4 线程/副本亦仅 6.73x） |
| 8 副本 × 16 路（threads=3） | 5.60x | 1093ms | 线程怎么配都无效 |
| 6 副本 × 8 路（realtime） | 6.36x | — | late 98%，崩溃 |
| 4 副本 × 16 路（realtime） | 6.07x | — | late 98%，崩溃 |

**结论**：进程内多副本受 GIL 与共享 torch OMP 线程池限制，不能复刻多进程扩展——V2 的池化价值在自愈重载/会话绑定/故障隔离，容量上与 V1 等价（单进程 ~8x、realtime ~4 路），扩容仍需多进程；进程内扩容唯一实测有效的路径是合批（方案 C）。

> 📄 图文混排四版本对比报告（含 V1/V2/V3/V4 全部数据与图表）：**docs/benchmark_report_pass1_versions_2026-09-04.pdf**

## 4. 资源与功能对比

| 维度 | 原版（torch） | 方案 C（ONNX int8） |
|---|---:|---:|
| 进程常驻内存（干净进程） | ~1152 MB（峰值，含加载瞬态） | **~155 MB**（int8 图磁盘 82MB；磁盘 284.6→81.8MB 为 int8 量化 3.4x） |
| 模型加载时间 | ~9 s | **~1.1 s** |
| 单进程并发流上限（按时） | 4 路（8 路临界） | **16 路** |
| 一致性保障 | 基线 | 与官方单流参考逐句一致；确定性输出（dither=0） |
| 依赖 | funasr | + onnxruntime / funasr-onnx（可选依赖，已注明） |
| 启用方式 | 现状 | `streaming_asr.mode="onnx"`（默认仍 replica，可灰度） |

容量换算（24 vCPU 单机，按引擎级实测保守外推）：
- 原版多实例：**24 路实时流/机**（6 实例×4 路，9/3 网关级验证）→ 100 路 ≈ 4~5 台。
- 方案 C：单进程 16 路按时；2 进程 46.6x offline → **3~4 进程 ≈ 48~64 路按时、100 路需 2 台以内**（网关级复测建议见 §6）。

---

## 5. 过程中的关键工程发现（对后续维护重要）

1. **decoder FSMN 零填充污染**（B/C 共有）：合批时 CIF 触发数不同的行被零填充到 max，FSNM 记忆把 padding 零帧卷进去 → 后续双字/丢字。修复 = 按 token 数分组执行 decoder；0-token 组跳过（单流参考语义）。
2. **前端默认 dither=1.0**：两套官方前端默认给波形加随机抖动，同音频两次运行输出不同——会掩盖真 bug、伪造测试结论。方案 C 已固定 dither=0（官方 C++ runtime 同款默认）。
3. **chunk_size 配置敏感性在 ONNX 路径同样成立**：默认配置 `chunk_size=[0,10,5]` 配小参数模型产生确定性劣化文本（网关冒烟实测）；启用 onnx 模式必须确保 chunk_size 与导出模型匹配（小模型 [5,10,5]）。
4. **torch 张量层合批（方案 B）教训**：数学可行（位级一致已证明）≠ 工程值得——funasr 内部无文档语义（(1,B) cache 回写、attention cache 时间维随历史变化等）踩坑成本高，且 CPU 上性能倒退。

## 6. 结论与建议

1. **合入 `feat/streaming-onnx` 作为首遍扩容主路径**：`streaming_asr.mode="onnx"` 一开关启用，默认 replica 保持不变（灰度安全）；部署前置 `scripts/export_onnx.py` 导出模型（int8 推荐，追求精度可 fp32 仍有 2~3x 提升）。
2. **网关级复测**：用 `benchmark/bench_gateway.py`（final=mock 隔离二遍）复跑 1/2/4 实例 × 8/16/32 会话矩阵，验证端到端容量与首 partial 延迟（引擎级 16 路按时为保守下界）。
3. **int8 精度抽检**：本次 4 段真实音频 CER=0（样本有限），建议用业务音频抽检后再全量切 int8。
4. **后续调优**：batch_size 16~32、batch_window 10~15ms；VAD（FSMN）可套用同一合批模式进一步降低排队。
5. `feat/streaming-batched`（方案 B）正确性已完全证明，但因性能低于基线，保留为研究记录，不合入。

## 7. 复现命令

```bash
# 原版（主仓）
.venv/bin/python benchmark/bench_paraformer.py --pacing offline,realtime --procs 1 \
    --streams 1,4,8 --threads 8 --out_dir benchmark/results/batch_compare
# 方案 C（worktree ../asrflow-onnx）
../asrflow/.venv/bin/python benchmark/bench_paraformer.py --engine onnx --onnx_dir models/onnx \
    --pacing offline,realtime --procs 1 --streams 1,4,8,16 --threads 8 --out_dir benchmark/results/batch_compare
# 方案 C 多进程
... --engine onnx --procs 2 --streams 8 ...
# 方案 B（worktree ../asrflow-batched）
../asrflow/.venv/bin/python benchmark/bench_paraformer.py --engine batched --pacing offline \
    --procs 1 --streams 1,4,8 --threads 8 --out_dir benchmark/results/batch_compare
# 一致性门禁 / 网关冒烟（worktree ../asrflow-onnx）
../asrflow/.venv/bin/python scripts/test_onnx_consistency.py --streams 6 --seg_sec 20
../asrflow/.venv/bin/python scripts/test_onnx_consistency.py --int8
../asrflow/.venv/bin/python scripts/smoke_onnx_gateway.py
```
