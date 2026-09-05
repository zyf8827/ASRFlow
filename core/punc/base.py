from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BasePuncEngine(ABC):
    """
    Abstract interface for Punctuation Engine (CT-Punc).
    """

    @abstractmethod
    def add_punctuation(
        self, text: str, cache: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add punctuation to raw Chinese text.
        """
        pass
