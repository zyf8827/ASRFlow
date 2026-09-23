from loguru import logger
from config.settings import PuncConfig
from core.punc.base import BasePuncEngine
from core.punc.mock_punc import MockPuncEngine


def create_punc_engine(config: PuncConfig) -> BasePuncEngine:
    """
    backend:
      - "mock": tests / explicit stub only
      - "funasr": real CT-Punc; hard error if unavailable
      - "auto": same as funasr — real engine or fail loudly (no silent Mock)
    """
    backend = config.backend.lower()
    if backend == "mock":
        logger.info("[PuncFactory] selected backend=mock (explicit)")
        return MockPuncEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.punc.ct_punc import CTPuncEngine

            engine = CTPuncEngine(config)
            logger.info(f"[PuncFactory] selected backend={backend} -> funasr")
            return engine
        except Exception as e:
            logger.error(
                f"[PuncFactory] CT-Punc unavailable for backend={backend!r}: {e}. "
                "Set punc.backend=mock only for tests; auto/funasr require models."
            )
            raise RuntimeError(
                f"Punc backend {backend!r} requires FunASR CT-Punc; "
                f"refusing silent Mock fallback. Underlying error: {e}"
            ) from e

    raise ValueError(f"Unsupported Punc backend: {config.backend}")


__all__ = ["BasePuncEngine", "MockPuncEngine", "create_punc_engine"]
