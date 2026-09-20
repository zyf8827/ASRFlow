import os
try:
    import yaml
except ImportError:
    try:
        from ruamel import yaml
    except ImportError:
        yaml = None

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def detect_default_device() -> str:
    """
    Auto-detect the most suitable default device:
    - If NVIDIA CUDA is available -> 'cuda:0'
    - Else if Ascend NPU / CPU -> 'cpu' (with vLLM on NPU/GPU)
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 10095
    http_port: int = 10096
    max_connections: int = 100
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    # Grace period for resuming a session after an unexpected disconnect.
    # Within this window the client may reconnect with {"type":"start",
    # "session_id":..., "resume":true} and continue the same session
    # (audio gap must be replayed by the client). 0 disables resume.
    resume_ttl_sec: float = 30.0
    # Idle eviction timeout for active sessions without any WS activity
    idle_timeout_sec: float = 300.0
    # Max duration (ms) of a single binary audio frame; longer frames are
    # truncated so one message can never occupy the session loop for minutes
    # (WS max_message_size alone allows ~312s). (env ASR_MAX_FRAME_MS)
    max_audio_frame_ms: int = 2000


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bytes_per_sample: int = 2  # PCM16
    ring_buffer_duration_sec: float = 32.0  # >= max_speech + pre_roll (30s + 0.8s)
    pre_roll_ms: int = 800  # Pre-roll window to prepend before VAD speech_start (500~1000ms)

    @property
    def bytes_per_ms(self) -> int:
        return (self.sample_rate * self.channels * self.bytes_per_sample) // 1000

    @property
    def pre_roll_bytes(self) -> int:
        return self.pre_roll_ms * self.bytes_per_ms


@dataclass
class VoiceConfig:
    """Streaming hallucination / degenerate-row layered defense (L0/L1/L2).

    See docs/design_degenerate_rows_guard_2026-09-18.md. Independent switches
    so each layer can roll back without the others.
    """
    # L1: do not emit partials until VAD speech start. Accumulation is
    # unconditional so pre-start tokens ride out on the first in-speech partial.
    # (env VOICE_SUPPRESS_PRESTART_PARTIALS; default on, P1)
    suppress_prestart_partials: bool = True
    # L0: drop streaming tokens when not in speech and chunk RMS is below this
    # (dBFS). -80 only catches digital silence; analog noise at -70 is untouched.
    # (env VOICE_SILENCE_FLOOR_DBFS)
    silence_floor_dbfs: float = -80.0
    # L0 optional: skip streaming inference on near-zero chunks (default still
    # runs inference so feats/cache stay on the timeline).
    # (env VOICE_SKIP_INFERENCE_ON_SILENCE)
    skip_inference_on_silence: bool = False
    # L2 watchdog: when non-speech accumulation reaches min_chars, force one
    # Qwen+Guard adjudication of the trailing segment (bounded by the hourly
    # rate limit) instead of guessing locally. Default on: analog-noise devices
    # bypass L0 and would otherwise grow current_partial_text unbounded on
    # 7-12h ABORT sessions. (env VOICE_WATCHDOG_ENABLE)
    watchdog_enable: bool = True
    watchdog_min_chars: int = 8  # env VOICE_WATCHDOG_MIN_CHARS
    # Max audio span pulled per watchdog force-cut; must fit within ring buffer
    watchdog_max_segment_ms: int = 30000
    watchdog_rate_limit_per_hour: int = 20


@dataclass
class WorkerPoolConfig:
    # CPU 推理线程池大小 (VAD/流式ASR/说话人共用)。0 = 自动: min(16, CPU 逻辑
    # 核数) —— 多实例绑核部署下单实例不宜固定 16 线程 (env ASR_WORKER_THREADS)
    worker_threads: int = 0


@dataclass
class AdmissionConfig:
    """负载准入控制: 探针 EWMA 持续超限 -> 拒绝新建会话 (滞回状态机)。

    过载模式从"接受连接然后无限劣化"变为"新会话收到 error 4290 + close 1013,
    存量会话不受影响"。resume 重连已持有容量, 不经过准入。
    """
    enable: bool = True  # env ADMISSION_ENABLE, 一键回滚开关
    # 探针阈值: 首遍合批排队 EWMA(ms) / VAD 锁排队 EWMA(ms) / 二遍回退率 EWMA
    # (env ADMISSION_STREAMING_WAIT_MS / ADMISSION_VAD_WAIT_MS /
    #  ADMISSION_FINAL_FALLBACK_RATE)
    streaming_wait_ms_limit: float = 200.0
    vad_wait_ms_limit: float = 150.0
    final_fallback_rate_limit: float = 0.5
    # 滞回: 任一探针持续超限 saturate_sec 进入饱和; 全部低于半阈值持续
    # recover_sec 才解除 (不对称阈值防抖动)
    saturate_sec: float = 5.0
    recover_sec: float = 10.0


@dataclass
class StreamingASRConfig:
    backend: str = "auto"  # auto | onnx | mock; onnx 为唯一真实引擎, auto 缺依赖时回退 Mock
    device: str = "auto"  # auto | cuda:0 | cpu
    # 流式重叠窗 [左, 当前, 右] (LFR chunk 数), 须与导出 ONNX 图的模型一致;
    # 小参数 Paraformer 导出物必须带左上下文 [5,10,5] (env STREAMING_CHUNK_SIZE)
    chunk_size: List[int] = field(default_factory=lambda: [5, 10, 5])
    # max streams coalesced into one forward (env STREAMING_BATCH_SIZE)
    batch_size: int = 8
    # window (ms) to wait while filling a batch (env STREAMING_BATCH_WINDOW_MS)
    batch_window_ms: int = 15
    # directory with funasr-exported model[_quant].onnx / decoder[_quant].onnx
    # + config.yaml / am.mvn / tokens.json (env STREAMING_ONNX_DIR)
    onnx_model_dir: str = "models/onnx"
    # use the int8-quantized graphs (smaller/faster, small accuracy delta)
    # (env STREAMING_ONNX_QUANT)
    onnx_quantize: bool = True
    # ORT intra-op threads per session (0 = ORT default = all cores)
    # (env STREAMING_ONNX_THREADS)
    onnx_intra_op_threads: int = 0


@dataclass
class VADConfig:
    backend: str = "auto"  # auto | funasr | mock
    model_name_or_path: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    model_revision: str = "v2.0.4"
    device: str = "cpu"  # FSMN-VAD is extremely lightweight on CPU
    max_end_silence_time: int = 800
    # Force-cut a sentence after this much continuous speech without a VAD
    # pause (ms). Keeps segments bounded for fluent/uninterrupted speech;
    # set <= 0 to disable.
    max_speech_duration_ms: int = 30000
    speech_noise_thresh: float = 0.6


@dataclass
class FinalASRConfig:
    engine_type: str = "vllm_http"  # vllm_http | openai_api | mock
    vllm_url: str = "http://127.0.0.1:8899/v1/audio/transcriptions"
    model_name: str = "qwen3-asr"
    api_key: str = "EMPTY"
    # 二遍单句硬超时(s)。压测全表最大延迟 442ms (32 并发/10s 段), 8s 有 ~18x
    # 余量; 30s force-cut 长句外推 ~1.3s 仍有 6x 余量, 2.5s 会造成长句系统性
    # 回退首遍 (env FINAL_TIMEOUT_SEC)
    hard_timeout_sec: float = 8.0
    max_batch_size: int = 8
    batch_window_ms: int = 20  # Micro-batch aggregation window (10~30ms)
    # 二遍并发上限, 按句计 (同时在途的 HTTP 请求数)。压测最高档 32 并发 p99
    # 仍 <450ms 且未见饱和拐点 (540x 实时), 取实测上沿 (env FINAL_MAX_CONCURRENCY)
    max_concurrency: int = 32
    # Bounded pending-job queue; overflow rejects the submit (pipeline falls
    # back to the provisional text) instead of piling up segment audio in RAM.
    max_queue_size: int = 64
    temperature: float = 0.0
    # 单句生成 token 上限: 只是截断上限不是固定开销。正常 30s 语速 ~140 字
    # (~120 token), 快语速 (护栏上限 12 字/s) 30s 可达 360 字, 256 会截断
    # (env FINAL_MAX_TOKENS)
    max_tokens: int = 512
    prompt_prefix: str = "语音转写："
    # 二遍 prompt 历史上下文尾部字数 (S3)。传已 finalize 的二遍结果尾部,
    # 不传本句首遍 (首遍错字会污染二遍)。0 = 不传 context (S1)。
    # (env FINAL_CONTEXT_HISTORY_CHARS)
    context_history_chars: int = 200


@dataclass
class SpeakerConfig:
    backend: str = "auto"  # auto | funasr | mock
    model_name_or_path: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    model_revision: str = "master"
    device: str = "auto"  # auto | cuda:0 | cpu
    similarity_threshold: float = 0.65
    max_speakers: int = 10


@dataclass
class PuncConfig:
    enable_realtime: bool = False
    backend: str = "auto"  # auto | funasr | mock
    model_name_or_path: str = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    model_revision: str = "v2.0.4"
    device: str = "cpu"


@dataclass
class GuardConfig:
    # 护栏只标记(needs_review)不换文: 首遍质量显著差于二遍, 异常时保留二遍文本;
    # 回退首遍仅发生在管线异常分支(二遍超时/失败/队列溢出, 无二遍输出可用)
    enable: bool = True
    max_edit_distance_ratio: float = 0.60  # Ratio above which revision is flagged for review
    max_chars_per_second: float = 12.0  # Human speech rarely exceeds 10-12 chars/sec
    min_speech_duration_ms: int = 300  # Audio shorter than this with long text is suspicious
    repetition_ngram: int = 4
    repetition_max_count: int = 3


@dataclass
class ITNConfig:
    enable: bool = True


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    # Directory for log files (relative paths resolve against the project
    # root, independent of the working directory). null disables file logging.
    log_dir: Optional[str] = "logs"
    # Explicit full path of the log file; when set it takes precedence over
    # log_dir. null falls back to <log_dir>/asrflow.log.
    log_file: Optional[str] = None
    # Rotation policy (loguru syntax): "00:00" = daily at midnight,
    # "1 day" = every 24h, "100 MB" = by size. null/empty disables rotation.
    log_rotation: Optional[str] = "00:00"
    # Retention of rotated files (loguru syntax), e.g. "30 days", "7 days".
    # null/empty keeps files forever.
    log_retention: Optional[str] = "30 days"
    # Directory for persisting per-session transcript JSON files on session
    # removal (durable record against disconnects). null disables.
    transcript_dir: Optional[str] = None


@dataclass
class RegistryConfig:
    """Service registration (Nacos). Registers the WS service port as an
    ephemeral instance with heartbeat, for client/service discovery."""
    enable: bool = False
    server_endpoint: str = "127.0.0.1:8848"  # Nacos server host:port
    service_name: str = "asrflow"
    # Address registered into Nacos; must be reachable by callers (NOT 0.0.0.0).
    # In containers / multi-NIC hosts set it explicitly (env NACOS_SERVICE_HOST).
    service_host: str = "127.0.0.1"
    namespace: str = "public"
    group_name: str = "DEFAULT_GROUP"
    username: str = ""
    password: str = ""
    heartbeat_sec: float = 5.0
    cluster_name: str = "DEFAULT"


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    pool: WorkerPoolConfig = field(default_factory=WorkerPoolConfig)
    streaming_asr: StreamingASRConfig = field(default_factory=StreamingASRConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    final_asr: FinalASRConfig = field(default_factory=FinalASRConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    punc: PuncConfig = field(default_factory=PuncConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    itn: ITNConfig = field(default_factory=ITNConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)

    def validate(self):
        """Validate cross-field invariants; raise ValueError on violation."""
        # max_speech_duration must fit within ring buffer
        if self.vad.max_speech_duration_ms > 0:
            budget_ms = self.audio.ring_buffer_duration_sec * 1000
            needed_ms = self.vad.max_speech_duration_ms + self.audio.pre_roll_ms
            if needed_ms > budget_ms:
                raise ValueError(
                    f"vad.max_speech_duration_ms ({self.vad.max_speech_duration_ms}) + "
                    f"audio.pre_roll_ms ({self.audio.pre_roll_ms}) = {needed_ms}ms exceeds "
                    f"audio.ring_buffer_duration_sec ({self.audio.ring_buffer_duration_sec}s = {budget_ms}ms). "
                    f"Increase ring_buffer_duration_sec or reduce max_speech_duration_ms."
                )
        # watchdog force-cut segment must fit within ring buffer
        if self.voice.watchdog_enable:
            budget_ms = self.audio.ring_buffer_duration_sec * 1000
            needed_ms = self.voice.watchdog_max_segment_ms + self.audio.pre_roll_ms
            if needed_ms > budget_ms:
                raise ValueError(
                    f"voice.watchdog_max_segment_ms ({self.voice.watchdog_max_segment_ms}) + "
                    f"audio.pre_roll_ms ({self.audio.pre_roll_ms}) = {needed_ms}ms exceeds "
                    f"audio.ring_buffer_duration_sec ({self.audio.ring_buffer_duration_sec}s = {budget_ms}ms). "
                    f"Increase ring_buffer_duration_sec or reduce watchdog_max_segment_ms."
                )

    def resolve_devices(self, primary_device: Optional[str] = None):
        """Resolve auto device bindings to concrete device strings."""
        dev = primary_device if primary_device and primary_device != "auto" else detect_default_device()

        if self.streaming_asr.device == "auto":
            self.streaming_asr.device = dev
        if self.speaker.device == "auto":
            self.speaker.device = dev
        if self.pool.worker_threads <= 0:
            self.pool.worker_threads = min(16, os.cpu_count() or 4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        config = cls()
        if not data:
            config.resolve_devices()
            try:
                config.validate()
            except ValueError as e:
                from loguru import logger

                logger.error(f"[AppConfig] Validation failed: {e}")
                raise
            return config

        def update_dataclass(target, src_dict, prefix=""):
            if not isinstance(src_dict, dict):
                return
            for key, val in src_dict.items():
                if hasattr(target, key):
                    attr = getattr(target, key)
                    if hasattr(attr, "__dataclass_fields__") and isinstance(val, dict):
                        update_dataclass(attr, val, prefix=f"{prefix}{key}.")
                    else:
                        setattr(target, key, val)
                else:
                    from loguru import logger

                    logger.warning(f"[AppConfig] Ignoring unknown config key {prefix}{key}")

        for section_name, section_dict in data.items():
            if hasattr(config, section_name) and isinstance(section_dict, dict):
                target_section = getattr(config, section_name)
                update_dataclass(target_section, section_dict, prefix=f"{section_name}.")
            else:
                from loguru import logger

                logger.warning(f"[AppConfig] Ignoring unknown config section '{section_name}'")

        config.resolve_devices()
        try:
            config.validate()
        except ValueError as e:
            from loguru import logger

            logger.error(f"[AppConfig] Validation failed: {e}")
            raise
        return config

    @classmethod
    def load_from_yaml(cls, path: str) -> "AppConfig":
        if os.path.exists(path):
            try:
                if yaml is None:
                    raise RuntimeError("No YAML parser available (install pyyaml)")
                if hasattr(yaml, "safe_load"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                elif hasattr(yaml, "YAML"):
                    yml = yaml.YAML(typ="safe")
                    with open(path, "r", encoding="utf-8") as f:
                        data = yml.load(f) or {}
                else:
                    data = {}
                return cls.from_dict(dict(data))
            except Exception as e:
                from loguru import logger

                logger.error(f"[AppConfig] Failed to load config from {path}: {e}")
                raise
        return cls()

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        path = config_path or os.getenv("ASR_CONFIG_PATH")
        if not path:
            default_path = os.path.join(os.path.dirname(__file__), "config.default.yaml")
            if os.path.exists(default_path):
                path = default_path

        if path and os.path.exists(path):
            config = cls.load_from_yaml(path)
        else:
            config = cls()

        # Environment variable overrides
        if os.getenv("ASR_DEVICE"):
            dev = os.getenv("ASR_DEVICE")
            config.streaming_asr.device = dev
            config.speaker.device = dev
        if os.getenv("ASR_SERVER_HOST"):
            config.server.host = os.getenv("ASR_SERVER_HOST")
        if os.getenv("ASR_SERVER_PORT"):
            config.server.port = int(os.getenv("ASR_SERVER_PORT"))
        if os.getenv("ASR_HTTP_PORT"):
            config.server.http_port = int(os.getenv("ASR_HTTP_PORT"))
        if os.getenv("VLLM_URL"):
            config.final_asr.vllm_url = os.getenv("VLLM_URL")
        if os.getenv("STREAMING_BATCH_SIZE"):
            config.streaming_asr.batch_size = int(os.getenv("STREAMING_BATCH_SIZE"))
        if os.getenv("STREAMING_BATCH_WINDOW_MS"):
            config.streaming_asr.batch_window_ms = int(os.getenv("STREAMING_BATCH_WINDOW_MS"))
        if os.getenv("STREAMING_ONNX_DIR"):
            config.streaming_asr.onnx_model_dir = os.getenv("STREAMING_ONNX_DIR")
        if os.getenv("STREAMING_ONNX_QUANT"):
            config.streaming_asr.onnx_quantize = os.getenv("STREAMING_ONNX_QUANT").strip().lower() in ("1", "true", "yes", "on")
        if os.getenv("STREAMING_ONNX_THREADS"):
            config.streaming_asr.onnx_intra_op_threads = int(os.getenv("STREAMING_ONNX_THREADS"))
        if os.getenv("STREAMING_CHUNK_SIZE"):
            # "5,10,5" or "5 10 5" -> [5, 10, 5]; the small (non-large) streaming
            # Paraformer requires a left context, i.e. chunk_size[0] = 5
            config.streaming_asr.chunk_size = [
                int(x) for x in os.getenv("STREAMING_CHUNK_SIZE").replace(",", " ").split()
            ]
        if os.getenv("FINAL_ASR_MODEL"):
            config.final_asr.model_name = os.getenv("FINAL_ASR_MODEL")
        if os.getenv("VAD_MODEL"):
            config.vad.model_name_or_path = os.getenv("VAD_MODEL")
        if os.getenv("SPEAKER_MODEL"):
            config.speaker.model_name_or_path = os.getenv("SPEAKER_MODEL")
        if os.getenv("PUNC_MODEL"):
            config.punc.model_name_or_path = os.getenv("PUNC_MODEL")
        if os.getenv("VAD_MAX_SPEECH_MS"):
            config.vad.max_speech_duration_ms = int(os.getenv("VAD_MAX_SPEECH_MS"))
        if os.getenv("RESUME_TTL_SEC"):
            config.server.resume_ttl_sec = float(os.getenv("RESUME_TTL_SEC"))
        if os.getenv("IDLE_TIMEOUT_SEC"):
            config.server.idle_timeout_sec = float(os.getenv("IDLE_TIMEOUT_SEC"))
        if os.getenv("ASR_MAX_FRAME_MS"):
            config.server.max_audio_frame_ms = int(os.getenv("ASR_MAX_FRAME_MS"))
        if os.getenv("MAX_CONNECTIONS"):
            config.server.max_connections = int(os.getenv("MAX_CONNECTIONS"))
        if os.getenv("ASR_WORKER_THREADS"):
            config.pool.worker_threads = int(os.getenv("ASR_WORKER_THREADS"))

        # Final ASR tuning overrides
        if os.getenv("FINAL_TIMEOUT_SEC"):
            config.final_asr.hard_timeout_sec = float(os.getenv("FINAL_TIMEOUT_SEC"))
        if os.getenv("FINAL_MAX_CONCURRENCY"):
            config.final_asr.max_concurrency = int(os.getenv("FINAL_MAX_CONCURRENCY"))
        if os.getenv("FINAL_MAX_TOKENS"):
            config.final_asr.max_tokens = int(os.getenv("FINAL_MAX_TOKENS"))
        if os.getenv("FINAL_CONTEXT_HISTORY_CHARS"):
            config.final_asr.context_history_chars = int(os.getenv("FINAL_CONTEXT_HISTORY_CHARS"))

        # Admission control overrides
        if os.getenv("ADMISSION_ENABLE"):
            config.admission.enable = os.getenv("ADMISSION_ENABLE").strip().lower() in ("1", "true", "yes", "on")
        if os.getenv("ADMISSION_STREAMING_WAIT_MS"):
            config.admission.streaming_wait_ms_limit = float(os.getenv("ADMISSION_STREAMING_WAIT_MS"))
        if os.getenv("ADMISSION_VAD_WAIT_MS"):
            config.admission.vad_wait_ms_limit = float(os.getenv("ADMISSION_VAD_WAIT_MS"))
        if os.getenv("ADMISSION_FINAL_FALLBACK_RATE"):
            config.admission.final_fallback_rate_limit = float(os.getenv("ADMISSION_FINAL_FALLBACK_RATE"))

        def _env_bool(name: str) -> Optional[bool]:
            raw = os.getenv(name)
            if raw is None:
                return None
            return raw.strip().lower() in ("1", "true", "yes", "on")

        flag = _env_bool("VOICE_SUPPRESS_PRESTART_PARTIALS")
        if flag is not None:
            config.voice.suppress_prestart_partials = flag
        if os.getenv("VOICE_SILENCE_FLOOR_DBFS"):
            config.voice.silence_floor_dbfs = float(os.getenv("VOICE_SILENCE_FLOOR_DBFS"))
        flag = _env_bool("VOICE_SKIP_INFERENCE_ON_SILENCE")
        if flag is not None:
            config.voice.skip_inference_on_silence = flag
        flag = _env_bool("VOICE_WATCHDOG_ENABLE")
        if flag is not None:
            config.voice.watchdog_enable = flag
        if os.getenv("VOICE_WATCHDOG_MIN_CHARS"):
            config.voice.watchdog_min_chars = int(os.getenv("VOICE_WATCHDOG_MIN_CHARS"))

        if os.getenv("FINAL_ASR_BACKEND"):
            config.final_asr.engine_type = os.getenv("FINAL_ASR_BACKEND")
        if os.getenv("STREAMING_ASR_BACKEND"):
            config.streaming_asr.backend = os.getenv("STREAMING_ASR_BACKEND")
        if os.getenv("VAD_BACKEND"):
            config.vad.backend = os.getenv("VAD_BACKEND")
        if os.getenv("SPEAKER_BACKEND"):
            config.speaker.backend = os.getenv("SPEAKER_BACKEND")
        if os.getenv("LOG_LEVEL"):
            config.observability.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_DIR"):
            config.observability.log_dir = os.getenv("LOG_DIR")
        if os.getenv("LOG_FILE"):
            config.observability.log_file = os.getenv("LOG_FILE")
        if os.getenv("LOG_ROTATION"):
            config.observability.log_rotation = os.getenv("LOG_ROTATION")
        if os.getenv("LOG_RETENTION"):
            config.observability.log_retention = os.getenv("LOG_RETENTION")

        # Nacos registry overrides
        if os.getenv("NACOS_ENABLE"):
            config.registry.enable = os.getenv("NACOS_ENABLE").strip().lower() in ("1", "true", "yes", "on")
        if os.getenv("NACOS_SERVER_ENDPOINT"):
            config.registry.server_endpoint = os.getenv("NACOS_SERVER_ENDPOINT")
        if os.getenv("NACOS_SERVICE_NAME"):
            config.registry.service_name = os.getenv("NACOS_SERVICE_NAME")
        if os.getenv("NACOS_SERVICE_HOST"):
            config.registry.service_host = os.getenv("NACOS_SERVICE_HOST")
        if os.getenv("NACOS_NAMESPACE"):
            config.registry.namespace = os.getenv("NACOS_NAMESPACE")
        if os.getenv("NACOS_GROUP_NAME"):
            config.registry.group_name = os.getenv("NACOS_GROUP_NAME")
        if os.getenv("NACOS_USERNAME"):
            config.registry.username = os.getenv("NACOS_USERNAME")
        if os.getenv("NACOS_PASSWORD"):
            config.registry.password = os.getenv("NACOS_PASSWORD")
        if os.getenv("NACOS_HEARTBEAT_SEC"):
            config.registry.heartbeat_sec = float(os.getenv("NACOS_HEARTBEAT_SEC"))

        config.resolve_devices()
        # Validate after all overrides
        config.validate()
        return config
