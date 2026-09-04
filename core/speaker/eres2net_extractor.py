import threading
from typing import Optional
import numpy as np
from loguru import logger

from core.speaker.base import BaseSpeakerEngine
from config.settings import SpeakerConfig


class ERes2NetSpeakerEngine(BaseSpeakerEngine):
    """
    FunASR Speaker Embedding Engine (ERes2NetV2 / CAM++).
    Runs on CPU to keep Ascend 910B dedicated to Qwen3-ASR Final.
    """

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from funasr import AutoModel

            logger.info(
                f"[ERes2NetSpeakerEngine] Loading speaker model '{self.config.model_name_or_path}' on {self.config.device}..."
            )
            self._model = AutoModel(
                model=self.config.model_name_or_path,
                model_revision=self.config.model_revision,
                device=self.config.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
            )
            logger.info("[ERes2NetSpeakerEngine] Speaker model loaded successfully.")
        except Exception as e:
            logger.error(f"[ERes2NetSpeakerEngine] Failed to load speaker model: {e}")
            raise

    def extract_embedding(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        if not audio_bytes or len(audio_bytes) < 3200:  # < 100ms
            return None

        if self._model is None:
            return None

        # Feed float32 samples, never raw bytes: funasr's load_bytes() sniffs
        # container magic on bytes input and can misdetect raw PCM16 segments
        # as MP3 (near-silence dither patterns), garbling the embedding input.
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self._lock:
            try:
                res = self._model.generate(input=samples)
                if not res or len(res) == 0:
                    return None
                spk_embedding = res[0].get("spk_embedding", None)
                if spk_embedding is None:
                    return None

                if hasattr(spk_embedding, "cpu"):
                    spk_embedding = spk_embedding.cpu().numpy()
                elif not isinstance(spk_embedding, np.ndarray):
                    spk_embedding = np.array(spk_embedding, dtype=np.float32)

                # Flatten and normalize L2
                vec = spk_embedding.flatten().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                return vec
            except Exception as e:
                logger.error(f"[ERes2NetSpeakerEngine] Embedding extraction failed: {e}")
                return None
