# AGENTS.md

异构 2-Pass 实时 ASR 网关（Paraformer 流式首遍 + 独立 vLLM 部署的 Qwen3-ASR 二遍）。

## 目录速览

- `core/` 引擎与会话核心（各引擎子目录 + session/session_manager/ring_buffer）；`server/` WebSocket/HTTP 网关与静态面板
- `pipeline/session_pipeline.py` 单会话处理链；`scripts/` 模型下载/ONNX 导出/e2e/健康检查/Demo 脚本；`config/` 默认配置；`fixtures/` 测试音频基准
- `deployment/` Docker 镜像与 compose 编排；`tests/` 全部基于 Mock 引擎
- `benchmark/` 压测套件（引擎级/网关级，用法见其 README）；`docs/` 技术文档与 `client-integration-guide.md`（客户端接入权威文档）
- `requirements-base.txt` 是仅网关+Mock 的最小依赖集；`requirements.txt` 为完整运行依赖

## 命令

- 环境：`pip install -r requirements-base.txt`（最小测试环境）或 `pip install -r requirements.txt`
- 测试（必须从仓库根目录运行）：`PYTHONPATH=. python3 -m unittest discover -s tests -v`
- 单个测试模块：`PYTHONPATH=. python3 -m unittest tests.test_ring_buffer -v`
- Mock 快速启动：`python3 main.py --port 10095 --http_port 10096 --streaming_backend mock --final_backend mock --vad_backend mock --speaker_backend mock`

## 关键约束

- **vllm 永远不进依赖**：Qwen3-ASR 由独立 vLLM Online Serving 进程部署，本服务仅通过 HTTP（`--vllm_url` / `VLLM_URL`）调用。requirements.txt 有意精简。URL 需以 `/v1/audio/transcriptions` 或 `/v1/chat/completions` 结尾；裸地址默认补全为 `/v1/audio/transcriptions`（见 `core/final_asr/qwen_engine.py`）。默认 `VLLM_URL` 为 `http://127.0.0.1:8899/v1/audio/transcriptions`，模型名 `qwen3-asr`。
- 所有导入以仓库根为顶层包（如 `from core.session import ...`），项目未安装为包——任何脚本/测试都必须在根目录下执行或带 `PYTHONPATH=.`。

## 架构要点

- 引擎工厂模式：每个组件 `core/<组件>/__init__.py` 提供 `create_*_engine(config)` 按 backend 字符串分发；新增后端需实现同目录 `base.py` 接口并在工厂注册。`backend="auto"` 与显式真实后端一样：真实引擎不可用则**启动失败**（拒绝静默 Mock）；仅显式 `backend="mock"` 用于测试。
- 首遍流式只有一个真实引擎：ONNX Runtime 多路合批（`core/streaming_asr/onnx_batched_streaming.py`，官方导出图 + int8 量化，模型目录 `models/onnx/` 由 `scripts/export_onnx.py` 产出）；backend 值 `auto|onnx|mock`。VAD/说话人支持 `funasr|mock|auto`，final 支持 `vllm_http|openai_api|mock`。
- 配置优先级：`config/config.default.yaml`（或 `ASR_CONFIG_PATH` 指定文件）< 环境变量（`ASR_DEVICE`、`VLLM_URL`、`FINAL_ASR_BACKEND` 等）< `main.py` CLI 参数。
- WebSocket 协议：Binary 帧 = PCM16/16kHz/单声道音频；Text 帧 = JSON 信令（start/commit/stop），网关层分离处理两种帧。完整协议（partial/provisional/final 消息、resume、热词）以 `docs/client-integration-guide.md` 为准，改协议须同步更新该文档。
- 默认端口：WS 10095，HTTP（healthz/metrics/sessions）10096；e2e 测试使用 10199/10198。

## 部署编排（deployment/）

- `init-compose.sh` 交互生成 `deployment/docker-compose.yaml` + `.env`，**两者均 gitignore 不入库**——不要直接改生成物，改 example 模板或脚本后重跑再生。single 模式=复制 `docker-compose.single.example.yaml`（含 sed 处理热替换注释行）；multi 模式=脚本按 nproc 内联生成绑核多实例。
- 容器日志默认挂宿主 `ASRFLOW_LOG_BASE`（默认 `/var/log/asrflow`），compose 里以 `${ASRFLOW_LOG_BASE:-默认}` 插值——改 `.env` 即换路径，无需重新生成。多实例按 `instance-N` 子目录区分互不冲突；`/app/logs` 是嵌套挂载，优先于源码热替换挂载 `..:/app`。
- 两份镜像**分开构建**：`build.sh` 产网关镜像 `asrflow:<版本>`，构建前逐文件校验 `models/` 完整性，缺文件即中止；网关 Dockerfile 按架构分两份（`Dockerfile`=amd64、`Dockerfile.arm64`=arm64），脚本按构建机架构自动选择，`PLATFORM=linux/arm64` 交叉构建、`CHECK_ONLY=1` 只校验不构建；`build-vllm.sh [gpu|ascend|ascend-310p]` 产二遍 Qwen3-ASR 镜像 `vllm-qwen3-asr:<vllm版本>[-ascend|-310p]`（gpu 用 `Dockerfile.vllm`：`vllm/vllm-openai:v0.28.0` 上用镜像自带 uv 补装；ascend 用 `Dockerfile.vllm.ascend`：`quay.io/ascend/vllm-ascend:v0.23.0` 上直接 pip 补装；ascend-310p 与 ascend 共用 `Dockerfile.vllm.ascend`，仅基础镜像换 `v0.23.0-310p`、tag 后缀 `-310p`；各 target 均为补装 `vllm[audio]` 同版本，pip 源统一阿里云——清华源在昇腾机实测拉不到 numpy 等候选，context 只用 `deployment/` 目录，不涉及 models 校验；运行编排参考 `docker-compose.qwen3-asr.gpu / .ascend .example.yaml` 两文件，310P 沿用 ascend 模板换镜像 tag）。

全部测试基于 Mock 引擎，无需 GPU、模型权重或 vLLM 实例即可运行。
