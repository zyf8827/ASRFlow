from loguru import logger
from config.settings import SpeakerConfig
from core.speaker.base import BaseSpeakerEngine
from core.speaker.mock_speaker import MockSpeakerEngine
from core.speaker.speaker_tracker import IncrementalSpeakerTracker
from core.speaker.recluster import global_recluster_speakers


def create_speaker_engine(config: SpeakerConfig) -> BaseSpeakerEngine:
    """
    backend:
      - "mock": tests / explicit stub only
      - "funasr": real speaker embedding; hard error if unavailable
      - "auto": same as funasr — real engine or fail loudly (no silent Mock)
    """
    backend = config.backend.lower()
    if backend == "mock":
        logger.info("[SpeakerFactory] selected backend=mock (explicit)")
        return MockSpeakerEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.speaker.eres2net_extractor import ERes2NetSpeakerEngine

            engine = ERes2NetSpeakerEngine(config)
            logger.info(f"[SpeakerFactory] selected backend={backend} -> funasr")
            return engine
        except Exception as e:
            logger.error(
                f"[SpeakerFactory] FunASR Speaker unavailable for backend={backend!r}: {e}. "
                "Set speaker.backend=mock only for tests; auto/funasr require models."
            )
            raise RuntimeError(
                f"Speaker backend {backend!r} requires FunASR speaker model; "
                f"refusing silent Mock fallback. Underlying error: {e}"
            ) from e

    raise ValueError(f"Unsupported Speaker backend: {config.backend}")


__all__ = [
    "BaseSpeakerEngine",
    "MockSpeakerEngine",
    "IncrementalSpeakerTracker",
    "global_recluster_speakers",
    "create_speaker_engine",
]
