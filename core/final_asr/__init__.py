from config.settings import FinalASRConfig
from core.final_asr.base import BaseFinalASREngine
from core.final_asr.qwen_engine import Qwen3ASREngine
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue


def create_final_asr_engine(config: FinalASRConfig) -> BaseFinalASREngine:
    engine_type = config.engine_type.lower()
    if engine_type == "mock":
        return MockQwenEngine(config)
    elif engine_type in ("vllm_http", "openai_api"):
        return Qwen3ASREngine(config)
    else:
        raise ValueError(f"Unsupported Final ASR engine type: {config.engine_type}")


__all__ = [
    "BaseFinalASREngine",
    "Qwen3ASREngine",
    "MockQwenEngine",
    "FinalQueue",
    "create_final_asr_engine",
]
