import threading
from typing import Optional, Dict, Any
from loguru import logger

from core.punc.base import BasePuncEngine
from config.settings import PuncConfig


class CTPuncEngine(BasePuncEngine):
    """
    FunASR CT-Transformer Real-time Punctuation Engine.
    """

    def __init__(self, config: PuncConfig):
        self.config = config
        self._lock = threading.Lock()
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from funasr import AutoModel

            logger.info(
                f"[CTPuncEngine] Loading CT-Punc model '{self.config.model_name_or_path}' on {self.config.device}..."
            )
            self._model = AutoModel(
                model=self.config.model_name_or_path,
                model_revision=self.config.model_revision,
                device=self.config.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
            )
            logger.info("[CTPuncEngine] CT-Punc loaded successfully.")
        except Exception as e:
            logger.error(f"[CTPuncEngine] Failed to load CT-Punc model: {e}")
            raise

    def add_punctuation(
        self, text: str, cache: Optional[Dict[str, Any]] = None
    ) -> str:
        if not text or not text.strip():
            return ""

        if self._model is None:
            return text

        kwargs = {}
        if cache is not None:
            kwargs["cache"] = cache

        with self._lock:
            try:
                res = self._model.generate(input=text, **kwargs)
                if not res or len(res) == 0:
                    return text
                punc_text = res[0].get("text", text)
                return punc_text.strip()
            except Exception as e:
                logger.error(f"[CTPuncEngine] Error during punctuation restoration: {e}")
                return text
