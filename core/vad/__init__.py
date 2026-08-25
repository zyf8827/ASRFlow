from loguru import logger
from config.settings import VADConfig
from core.vad.base import BaseVADEngine
from core.vad.mock_vad import MockVADEngine


def create_vad_engine(config: VADConfig) -> BaseVADEngine:
    """
    Create the VAD engine for this process (single FSMN-VAD model with
    self-healing; VAD is lightweight and needs no in-process pooling).
    """
    backend = config.backend.lower()
    if backend == "mock":
        return MockVADEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.vad.fsmn_vad import FSMNVADEngine

            return FSMNVADEngine(config)
        except Exception as e:
            if backend == "auto":
                logger.warning(
                    f"[VADFactory] FunASR VAD not available ({e}), falling back to MockVADEngine."
                )
                return MockVADEngine(config)
            raise e

    raise ValueError(f"Unsupported VAD backend: {config.backend}")


__all__ = ["BaseVADEngine", "MockVADEngine", "create_vad_engine"]
