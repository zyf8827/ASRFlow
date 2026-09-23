from loguru import logger
from config.settings import FinalASRConfig
from core.final_asr.base import BaseFinalASREngine
from core.final_asr.qwen_engine import Qwen3ASREngine
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue


def create_final_asr_engine(config: FinalASRConfig) -> BaseFinalASREngine:
    """
    engine_type:
      - "mock": tests / explicit stub only
      - "vllm_http" | "openai_api": real HTTP client (Qwen3ASREngine)
      - "auto": treated as vllm_http (never Mock)
    """
    engine_type = config.engine_type.lower()
    if engine_type == "mock":
        logger.info("[FinalASRFactory] selected engine_type=mock (explicit)")
        return MockQwenEngine(config)
    if engine_type in ("vllm_http", "openai_api", "auto"):
        resolved = "vllm_http" if engine_type == "auto" else engine_type
        logger.info(
            f"[FinalASRFactory] selected engine_type={engine_type} -> {resolved} "
            f"(vllm_url={config.vllm_url!r}, model={config.model_name!r})"
        )
        return Qwen3ASREngine(config)
    raise ValueError(f"Unsupported Final ASR engine type: {config.engine_type}")


__all__ = [
    "BaseFinalASREngine",
    "Qwen3ASREngine",
    "MockQwenEngine",
    "FinalQueue",
    "create_final_asr_engine",
]
