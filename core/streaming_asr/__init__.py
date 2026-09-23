from loguru import logger
from config.settings import StreamingASRConfig
from core.streaming_asr.base import BaseStreamingASREngine
from core.streaming_asr.mock_streaming import MockStreamingASREngine


def create_streaming_asr_engine(config: StreamingASRConfig) -> BaseStreamingASREngine:
    """
    Create the streaming ASR engine for this process.

    The only real engine is the ONNX Runtime multi-stream batched engine:
    one encoder/decoder session pair + one forward thread coalescing
    same-shape chunk forwards of multiple streams along the graphs'
    dynamic batch axis (官方导出图 + int8 量化).

    backend:
      - "mock": mock engine (tests / explicit local stub only)
      - "onnx": the real engine; hard error when it cannot be initialized
      - "auto": same as onnx — resolve to the real engine or fail loudly
        (no silent Mock fallback; product readiness must not look healthy
        on fake ASR)
    """
    backend = config.backend.lower()
    if backend == "mock":
        logger.info("[StreamingASRFactory] selected backend=mock (explicit)")
        return MockStreamingASREngine(config)

    if backend in ("auto", "onnx"):
        try:
            from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

            engine = OnnxBatchedStreamingEngine(config)
            logger.info(
                f"[StreamingASRFactory] selected backend={backend} -> onnx "
                f"(model_dir={config.onnx_model_dir!r})"
            )
            return engine
        except Exception as e:
            # auto and onnx both fail loudly — never silently mock in product
            logger.error(
                f"[StreamingASRFactory] ONNX engine unavailable for backend={backend!r}: {e}. "
                "Set streaming_asr.backend=mock only for tests; auto/onnx require a real model."
            )
            raise RuntimeError(
                f"Streaming ASR backend {backend!r} requires ONNX Runtime and models "
                f"under {config.onnx_model_dir!r}; refusing silent Mock fallback. "
                f"Underlying error: {e}"
            ) from e

    raise ValueError(f"Unsupported Streaming ASR backend: {config.backend}")


__all__ = [
    "BaseStreamingASREngine",
    "MockStreamingASREngine",
    "create_streaming_asr_engine",
]
