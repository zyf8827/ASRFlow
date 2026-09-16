import argparse
import asyncio
import os
import sys

# Ensure local packages in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConfig
from server.service import ASRRealtimeService


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASRFlow Heterogeneous 2-Pass Inference Service"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
    parser.add_argument("--host", type=str, default=None, help="Server host IP (e.g. 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="WebSocket Gateway port (default: 10095)")
    parser.add_argument("--http_port", type=int, default=None, help="HTTP API / Metrics port (default: 10096)")
    parser.add_argument("--vllm_url", type=str, default=None, help="vLLM endpoint URL")
    parser.add_argument(
        "--final_model",
        type=str,
        default=None,
        help="Pass-2 model name reported to the vLLM server "
        "(e.g. qwen3-asr-0.6b, must match its --served-model-name)",
    )
    parser.add_argument(
        "--vad_model",
        type=str,
        default=None,
        help="VAD model name or local directory path "
        "(local path skips ModelScope download)",
    )
    parser.add_argument(
        "--speaker_model",
        type=str,
        default=None,
        help="Speaker model name or local directory path",
    )
    parser.add_argument(
        "--punc_model",
        type=str,
        default=None,
        help="Realtime punctuation model name or local directory path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Primary execution device (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--streaming_backend",
        type=str,
        default=None,
        choices=["auto", "onnx", "mock"],
        help="Streaming ASR backend (onnx engine is the only real one; "
        "auto falls back to mock when unavailable)",
    )
    parser.add_argument(
        "--streaming_batch_size",
        type=int,
        default=None,
        help="onnx mode: max streams coalesced into one forward "
        "(env STREAMING_BATCH_SIZE)",
    )
    parser.add_argument(
        "--streaming_batch_window_ms",
        type=int,
        default=None,
        help="onnx mode: window (ms) to wait while filling a batch, 0 to "
        "execute on arrival (env STREAMING_BATCH_WINDOW_MS)",
    )
    parser.add_argument(
        "--streaming_onnx_dir",
        type=str,
        default=None,
        help="onnx mode: directory with the funasr-exported ONNX artifacts "
        "(env STREAMING_ONNX_DIR)",
    )
    parser.add_argument(
        "--streaming_onnx_fp32",
        action="store_true",
        default=None,
        help="onnx mode: use fp32 graphs instead of int8-quantized ones "
        "(env STREAMING_ONNX_QUANT=0 works too)",
    )
    parser.add_argument(
        "--final_backend",
        type=str,
        default=None,
        choices=["vllm_http", "openai_api", "mock"],
        help="Final ASR engine type",
    )
    parser.add_argument(
        "--vad_backend",
        type=str,
        default=None,
        choices=["auto", "funasr", "mock"],
        help="VAD backend",
    )
    parser.add_argument(
        "--speaker_backend",
        type=str,
        default=None,
        choices=["auto", "funasr", "mock"],
        help="Speaker backend",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for log files (relative to project root; null/omitted "
        "uses config default 'logs', empty string disables file logging) "
        "(env LOG_DIR)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Explicit log file path; takes precedence over --log_dir (env LOG_FILE)",
    )
    parser.add_argument(
        "--log_rotation",
        type=str,
        default=None,
        help='Rotation policy, loguru syntax: "00:00" daily at midnight, '
        '"1 day" every 24h, "100 MB" by size (env LOG_ROTATION)',
    )
    parser.add_argument(
        "--log_retention",
        type=str,
        default=None,
        help='Retention of rotated files, e.g. "30 days" (env LOG_RETENTION)',
    )
    parser.add_argument(
        "--max_speech_duration_ms",
        type=int,
        default=None,
        help="Force-cut a sentence after this much continuous speech without a "
        "VAD pause, in ms (keep <= the second-pass model's max audio length, "
        "e.g. ~30000 for typical Qwen3-ASR vLLM deployments; <=0 disables)",
    )
    parser.add_argument(
        "--resume_ttl_sec",
        type=float,
        default=None,
        help="Grace period for resuming a session after unexpected "
        "disconnect, in seconds (0 disables resume)",
    )
    parser.add_argument(
        "--idle_timeout_sec",
        type=float,
        default=None,
        help="Idle timeout for active sessions without WS activity, in seconds (env IDLE_TIMEOUT_SEC)",
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default=None,
        help="Directory to persist per-session transcript JSON files on "
        "session removal (empty/omitted disables)",
    )
    parser.add_argument(
        "--nacos_enable",
        action="store_true",
        default=None,
        help="Register this instance into Nacos for service discovery "
        "(env NACOS_ENABLE=true works too)",
    )
    parser.add_argument("--nacos_endpoint", type=str, default=None, help="Nacos server host:port (env NACOS_SERVER_ENDPOINT)")
    parser.add_argument("--nacos_service_name", type=str, default=None, help="Service name registered in Nacos (env NACOS_SERVICE_NAME)")
    parser.add_argument("--nacos_service_host", type=str, default=None, help="Service IP registered in Nacos, must NOT be 0.0.0.0 (env NACOS_SERVICE_HOST)")
    parser.add_argument("--nacos_namespace", type=str, default=None, help="Nacos namespace (env NACOS_NAMESPACE)")
    parser.add_argument("--nacos_group_name", type=str, default=None, help="Nacos group (env NACOS_GROUP_NAME)")
    parser.add_argument("--nacos_username", type=str, default=None, help="Nacos auth username (env NACOS_USERNAME)")
    parser.add_argument("--nacos_password", type=str, default=None, help="Nacos auth password (env NACOS_PASSWORD)")
    parser.add_argument("--nacos_heartbeat_sec", type=float, default=None, help="Nacos heartbeat interval in seconds (env NACOS_HEARTBEAT_SEC)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AppConfig.load(args.config)

    # CLI parameter overrides
    if args.device:
        config.resolve_devices(primary_device=args.device)

    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.http_port:
        config.server.http_port = args.http_port
    if args.vllm_url:
        config.final_asr.vllm_url = args.vllm_url
    if args.final_model:
        config.final_asr.model_name = args.final_model
    if args.vad_model:
        config.vad.model_name_or_path = args.vad_model
    if args.speaker_model:
        config.speaker.model_name_or_path = args.speaker_model
    if args.punc_model:
        config.punc.model_name_or_path = args.punc_model
    if args.streaming_backend:
        config.streaming_asr.backend = args.streaming_backend
    if args.streaming_batch_size is not None:
        config.streaming_asr.batch_size = args.streaming_batch_size
    if args.streaming_batch_window_ms is not None:
        config.streaming_asr.batch_window_ms = args.streaming_batch_window_ms
    if args.streaming_onnx_dir:
        config.streaming_asr.onnx_model_dir = args.streaming_onnx_dir
    if args.streaming_onnx_fp32:
        config.streaming_asr.onnx_quantize = False
    if args.final_backend:
        config.final_asr.engine_type = args.final_backend
    if args.vad_backend:
        config.vad.backend = args.vad_backend
    if args.speaker_backend:
        config.speaker.backend = args.speaker_backend
    if args.log_level:
        config.observability.log_level = args.log_level
    if args.log_dir is not None:
        config.observability.log_dir = args.log_dir or None
    if args.log_file:
        config.observability.log_file = args.log_file
    if args.log_rotation:
        config.observability.log_rotation = args.log_rotation
    if args.log_retention:
        config.observability.log_retention = args.log_retention
    if args.max_speech_duration_ms is not None:
        config.vad.max_speech_duration_ms = args.max_speech_duration_ms
    if args.resume_ttl_sec is not None:
        config.server.resume_ttl_sec = args.resume_ttl_sec
    if args.idle_timeout_sec is not None:
        config.server.idle_timeout_sec = args.idle_timeout_sec
    if args.transcript_dir:
        config.observability.transcript_dir = args.transcript_dir

    # Validate after CLI overrides and log effective config digest
    try:
        config.validate()
    except ValueError as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    # Log effective config summary for observability
    import logging

    logging.getLogger().info(
        f"Effective config: server={config.server.host}:{config.server.port}/{config.server.http_port} "
        f"pool={config.pool.worker_threads}t "
        f"streaming=onnx(batch={config.streaming_asr.batch_size},window={config.streaming_asr.batch_window_ms}ms) "
        f"vad.max_speech={config.vad.max_speech_duration_ms}ms ring={config.audio.ring_buffer_duration_sec}s "
        f"final(timeout={config.final_asr.hard_timeout_sec}s,conc={config.final_asr.max_concurrency}) "
        f"resume_ttl={config.server.resume_ttl_sec}s idle_timeout={config.server.idle_timeout_sec}s"
    )

    # Nacos registry overrides
    if args.nacos_enable:
        config.registry.enable = True
    if args.nacos_endpoint:
        config.registry.server_endpoint = args.nacos_endpoint
    if args.nacos_service_name:
        config.registry.service_name = args.nacos_service_name
    if args.nacos_service_host:
        config.registry.service_host = args.nacos_service_host
    if args.nacos_namespace:
        config.registry.namespace = args.nacos_namespace
    if args.nacos_group_name:
        config.registry.group_name = args.nacos_group_name
    if args.nacos_username:
        config.registry.username = args.nacos_username
    if args.nacos_password:
        config.registry.password = args.nacos_password
    if args.nacos_heartbeat_sec is not None:
        config.registry.heartbeat_sec = args.nacos_heartbeat_sec

    service = ASRRealtimeService(config)
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
