from typing import Optional, Dict, Any
from core.punc.base import BasePuncEngine
from config.settings import PuncConfig


class MockPuncEngine(BasePuncEngine):
    """
    Mock punctuation engine. Appends period or comma based on phrase length.
    """

    def __init__(self, config: PuncConfig):
        self.config = config

    def add_punctuation(
        self, text: str, cache: Optional[Dict[str, Any]] = None
    ) -> str:
        if not text:
            return ""
        trimmed = text.strip()
        if not trimmed.endswith(("。", "！", "？", "，", ".")):
            return trimmed + "。"
        return trimmed
