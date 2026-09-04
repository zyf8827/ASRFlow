# ASRFlow Benchmark Suite

本目录包含性能压测与评估工具，分两类：

| 脚本 | 层级 | 说明 |
| --- | --- | --- |
| `run_paraformer_suite.sh` | 套件 | **首遍 Paraformer（CPU）单模型压测**：引擎级多进程扩展性 + 网关级多实例负载均衡 |
| `run_qwen_suite.sh` | 套件 | **二遍 Qwen3-ASR（远程 vLLM/单 4090）单模型压测**：并发 1→N 递增加压 |
| `bench_paraformer.py` | 引擎级 | 直测 Paraformer 流式推理（绕过网关），多进程吞吐线性度 |
| `bench_qwen_vllm.py` | 引擎级 | 直连 vLLM transcriptions API（复用服务中的 `Qwen3ASREngine`），并发扫描 |
| `bench_gateway.py` | 网关级 | 真实 WS 协议多实例轮询负载均衡压测 |
| `benchmark.py` | 网关级 | （既有）合成音频多会话端到端压测 |

两个套件均以 `config/config.default.yaml` 为配置基准、`fixtures/sample_16k.wav` 为测试语音输入（支持通过 `AUDIO=/path/to/wav` 传入公开标准数据集如 AISHELL-1），
结果落盘 `benchmark/results/<时间戳>_<名>/`（CSV 指标 + JSON 明细 + 全程日志 + 主机 manifest），
可直接用于后续报告编写。

## 1. 首遍 Paraformer（CPU）单模型压测

```bash
bash benchmark/run_paraformer_suite.sh              # 全量（引擎级 + 网关级）
SKIP_GATEWAY=1 bash benchmark/run_paraformer_suite.sh   # 只跑引擎级
SKIP_ENGINE=1  bash benchmark/run_paraformer_suite.sh   # 只跑网关级
```

引擎级（`bench_paraformer.py`）回答两个问题：

* **单进程饱和点**：`process_chunk` 内部全局锁串行化推理，并发流共享一次推理。
  指标里的 `wait_ms_*`（锁等待）即排队延迟——并发加上去后 `wait` 涨、`call` 不涨，
  说明瓶颈在锁而非算力；`offline` 模式 `throughput_rtx` 为单进程纯算力实时倍率。
* **多进程线性度**：每进程独立加载模型，`paraformer_scaling.csv` 给出
  `speedup_vs_1proc` 与 `parallel_efficiency`（≥0.9 记为线性）。
  注意 `--threads` 固定（套件默认 2）才是公平的线性对比；`0=auto` 按核均分，
  适合模拟"整机切分给 N 个实例"的真实部署。

网关级（`bench_gateway.py`）验证多实例部署形态：N 个网关进程（`final_asr=mock`
隔离二遍、关闭 Nacos 注册），客户端按轮询分发会话，对比 N=1/2/4 的
首 partial 延迟与总吞吐（`gateway_all.csv` 已跨 N 拼接）。

常用覆盖（环境变量）：`PROCS_LIST`（进程数）、`STREAMS_LIST`（每进程流数）、
`THREADS`、`GATEWAY_PROCS`、`GATEWAY_SESSIONS`、`STREAM_SEC`、`GW_THREADS`。

## 2. 二遍 Qwen3-ASR（远程 vLLM，单 4090）单模型压测

```bash
bash benchmark/run_qwen_suite.sh
TIMEOUT=8 bash benchmark/run_qwen_suite.sh    # 测试 8s 硬超时口径下的超时率
```

* 并发从 1 逐步加到 32（`CONCURRENCY` 可调），段长 3s/10s 两种 VAD 断句形态；
* 指标：延迟分位数 p50/p90/p95/p99、`req_per_sec`、`audio_rtx`（可支撑的二遍实时流数）、
  错误分类（http/timeout/other）、`chars_*`（识别字符数 sanity 检查）；
* `TIMEOUT` 默认 30s 用于测真实容量曲线（避免把排队当超时）；设 8 则对齐配置
  `final_asr.hard_timeout_sec` 的回退口径；
* 每请求明细在 `qwen_vllm_details.json`，可画延迟分布图。

## 3. 核心指标定义

* **实时倍率 rtx / throughput_rtx**：处理音频秒数 ÷ 墙钟秒数；≥1 才能承载对应路数的实时流。
* **compute_rtf**：音频秒数 ÷ 纯推理时间（RTF 的倒数），去掉排队后的算力上限。
* **call_ms / wait_ms**（引擎级）：单次 `process_chunk` 总耗时 / 其中锁排队耗时。
  `streams=1` 时 `wait≈0`，`call` 即无竞争纯推理延迟（单路基线）。
* **first_partial_ms**（网关级）：首帧音频发出 → 首个非空 partial 返回。
* **end_to_end_ms**（网关级）：首帧音频发出 → `session_finished`。

## 4. 既有端到端压测（合成音频）

```bash
python3 benchmark/benchmark.py --uri ws://127.0.0.1:10095 --concurrency 10 --turns 3 \
  --speech_sec 4.0 --silence_sec 2.0 --output_json benchmark_report.json
```

## 5. 报告生成（图表 + PDF）

```bash
# 先装中文字体与工具 (一次性):
#   sudo apt-get install -y fonts-noto-cjk poppler-utils
#   uv pip install matplotlib weasyprint
# matplotlib 读 .ttc 只认首个字面(日文), 需抽取简中字面:
python3 - <<'EOF'
from fontTools.ttLib import TTCollection
import os, glob
for src in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"):
    tag = "Bold" if "Bold" in src else "Regular"
    for f in TTCollection(src).fonts:
        if (f["name"].getDebugName(1) or "") == "Noto Sans CJK SC":
            f.save(os.path.expanduser(f"~/.fonts/NotoSansCJKsc-{tag}.otf")); break
EOF

.venv/bin/python benchmark/make_report_charts.py <paraformer_suite目录> <qwen_suite目录> docs/benchmark_charts
.venv/bin/python benchmark/make_report_pdf.py   <paraformer_suite目录> <qwen_suite目录> docs/benchmark_charts docs/report.pdf
```

成品示例：`docs/benchmark_report_2026-09-03.pdf`（2 页 A4，4 图 + 结论卡片），
对应详版数据报告 `docs/benchmark_report_2026-09-03.md`。

## 6. 注意事项

* **预热与冷却**：所有脚本均带预热且预热不计入统计——引擎级每组合进程先预热
  5s（`--warmup_sec`）再同步起表计时；二遍每并发级别先发 2 个预热请求
  （`--warmup`）；网关级每实例先跑 1 轮 5s 短会话（`--warmup_sessions`/`--warmup_sec`）。
  级间冷却（`--cooldown`）等待 CPU/GPU 状态回落，避免上一档的热残留干扰下一档。
* 压测会占满 CPU/打满远程 GPU，仅在测试环境执行；网关级套件会临时占用
  21001+ / 21101+ 端口并自动启停服务进程。
* 首次运行 FunASR 需下载/加载模型（数分钟）；`results/` 目录已加入 .gitignore。
* 报告引用数据时，`manifest.json` 记录了主机（CPU 型号/核数/内存）、git 版本与完整
  压测参数，可直接溯源。
