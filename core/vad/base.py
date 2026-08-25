from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class BaseVADEngine(ABC):
    """
    Abstract interface for streaming Voice Activity Detection (VAD).
    """

    @abstractmethod
    def process_chunk(
        self, audio_bytes: bytes, cache: Dict[str, Any], is_final: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Process a chunk of audio bytes (PCM16 16kHz mono).
        Returns a list of detected speech segment boundaries [(start_ms, end_ms), ...].
        - start_ms != -1 indicates speech onset detected.
        - end_ms != -1 indicates speech offset / endpoint detected.
        Timestamps are milliseconds from the start of the current cache
        generation (both FunASR and Mock restart at 0 on an empty cache).
        The pipeline maps them onto the ring-buffer timeline.
        """
        pass
