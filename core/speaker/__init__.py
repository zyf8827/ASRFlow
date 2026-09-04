from loguru import logger
from config.settings import SpeakerConfig
from core.speaker.base import BaseSpeakerEngine
from core.speaker.mock_speaker import MockSpeakerEngine
from core.speaker.speaker_tracker import IncrementalSpeakerTracker
from core.speaker.recluster import global_recluster_speakers


def create_speaker_engine(config: SpeakerConfig) -> BaseSpeakerEngine:
    backend = config.backend.lower()
    if backend == "mock":
        return MockSpeakerEngine(config)

    if backend in ("auto", "funasr"):
        try:
            from core.speaker.eres2net_extractor import ERes2NetSpeakerEngine

            return ERes2NetSpeakerEngine(config)
        except Exception as e:
            if backend == "auto":
                logger.warning(
                    f"[SpeakerFactory] FunASR Speaker model not available ({e}), falling back to MockSpeakerEngine."
                )
                return MockSpeakerEngine(config)
            raise e

    raise ValueError(f"Unsupported Speaker backend: {config.backend}")


__all__ = [
    "BaseSpeakerEngine",
    "MockSpeakerEngine",
    "IncrementalSpeakerTracker",
    "global_recluster_speakers",
    "create_speaker_engine",
]
