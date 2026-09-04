from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseSpeakerEngine(ABC):
    """
    Abstract interface for Speaker Embedding Extraction (ERes2NetV2 / CAM++).
    """

    @abstractmethod
    def extract_embedding(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract speaker embedding vector (1D numpy array, float32) from a speech segment.
        """
        pass
