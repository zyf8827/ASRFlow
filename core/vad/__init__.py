from loguru import logger
from config.settings import VADConfig
from core.vad.base import BaseVADEngine
from core.vad.mock_vad import MockVADEngine


def create_vad_engine(config: VADConfig) -> BaseVADEngine:
    """
    Create the VAD engine for this process (single FSMN-VAD model with
    self-healing; VAD is lightweight and needs no in-process pooling).

    backend:
      - "mock": tests / explicit stub only
      - "funasr": real FSMN-VAD; hard error if unavailable
      - "auto": same as funasr — real engine or fail loudly (no silent Mock)
    """
    backend = config.backend.lower()
    if backend == "mock":
        logger.info("[VADFactory] selected backend=mock (explicit)")
        return MockVADEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.vad.fsmn_vad import FSMNVADEngine

            engine = FSMNVADEngine(config)
            logger.info(f"[VADFactory] selected backend={backend} -> funasr")
            return engine
        except Exception as e:
            logger.error(
                f"[VADFactory] FunASR VAD unavailable for backend={backend!r}: {e}. "
                "Set vad.backend=mock only for tests; auto/funasr require FunASR models."
            )
            raise RuntimeError(
                f"VAD backend {backend!r} requires FunASR FSMN-VAD; "
                f"refusing silent Mock fallback. Underlying error: {e}"
            ) from e

    raise ValueError(f"Unsupported VAD backend: {config.backend}")


__all__ = ["BaseVADEngine", "MockVADEngine", "create_vad_engine"]
