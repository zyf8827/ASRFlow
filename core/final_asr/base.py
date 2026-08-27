from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseFinalASREngine(ABC):
    """
    Abstract interface for Pass-2 Final ASR Engine (Qwen3-ASR).
    Takes a complete VAD speech segment and produces high-precision, punctuated final transcript.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        context: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        """
        Transcribe a single speech segment.
        """
        pass

    @abstractmethod
    async def transcribe_batch(
        self, batch_items: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Transcribe a batch of speech segments for micro-batching on Ascend NPU / vLLM.
        batch_items is a list of dicts: [{"audio_bytes": bytes, "context": str, "hotwords": list}, ...]
        Returns a list of transcribed texts.
        """
        pass
