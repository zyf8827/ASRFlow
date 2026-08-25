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
      - "mock": mock engine (tests)
      - "onnx": the real engine, hard error when it cannot be initialized
      - "auto": the real engine, silently falls back to mock when the ONNX
        runtime / model directory is unavailable (dependency-less dev env)
    """
    backend = config.backend.lower()
    if backend == "mock":
        return MockStreamingASREngine(config)

    if backend in ("auto", "onnx"):
        try:
            from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

            return OnnxBatchedStreamingEngine(config)
        except Exception as e:
            if backend == "auto":
                logger.warning(
                    f"[StreamingASRFactory] ONNX engine not available ({e}), "
                    "falling back to MockStreamingASREngine."
                )
                return MockStreamingASREngine(config)
            raise

    raise ValueError(f"Unsupported Streaming ASR backend: {config.backend}")


__all__ = [
    "BaseStreamingASREngine",
    "MockStreamingASREngine",
    "create_streaming_asr_engine",
]
