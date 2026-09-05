from loguru import logger
from config.settings import PuncConfig
from core.punc.base import BasePuncEngine
from core.punc.mock_punc import MockPuncEngine


def create_punc_engine(config: PuncConfig) -> BasePuncEngine:
    backend = config.backend.lower()
    if backend == "mock":
        return MockPuncEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.punc.ct_punc import CTPuncEngine

            return CTPuncEngine(config)
        except Exception as e:
            if backend == "auto":
                logger.warning(
                    f"[PuncFactory] CT-Punc model not available ({e}), falling back to MockPuncEngine."
                )
                return MockPuncEngine(config)
            raise e

    raise ValueError(f"Unsupported Punc backend: {config.backend}")


__all__ = ["BasePuncEngine", "MockPuncEngine", "create_punc_engine"]
