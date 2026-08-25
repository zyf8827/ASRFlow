from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseStreamingASREngine(ABC):
    """
    Abstract interface for Streaming 1-Pass ASR Engine (ONNX batched).
    """

    @abstractmethod
    def process_chunk(
        self,
        audio_bytes: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        """
        Process a streaming audio chunk and return the current partial/provisional recognized text.
        """
        pass

    @abstractmethod
    def flush(
        self, cache: Dict[str, Any], hotwords: Optional[List[str]] = None
    ) -> str:
        """
        Flush model state at VAD sentence boundary with is_final=True to produce provisional text.
        """
        pass
