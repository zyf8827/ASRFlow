import threading
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from loguru import logger

from core.vad.base import BaseVADEngine
from core.engine_health import SelfHealingMixin
from core.metrics.prometheus_metrics import metrics
from config.settings import VADConfig


class FSMNVADEngine(SelfHealingMixin, BaseVADEngine):
    """
    FSMN-VAD streaming engine powered by FunASR AutoModel.
    CPU-efficient and optimized for real-time speech boundary detection.

    Self-healing (via SelfHealingMixin): consecutive inference failures unload
    the model and reload it in the background; calls while it is down return
    [] (no boundaries), the same degradation as a single transient error. VAD
    state lives in the per-session cache dict, which stays valid across a
    reload of the identical model, so bound sessions resume seamlessly.
    """

    METRIC_ERRORS = "asr_vad_engine_errors"
    METRIC_RELOADS = "asr_vad_engine_reloads"
    METRIC_RELOADING_GAUGE = "asr_vad_engine_reloading"

    def __init__(
        self,
        config: VADConfig,
        name: Optional[str] = None,
        reload_after_errors: int = 3,
    ):
        self.config = config
        self.name = name
        self._lock = threading.Lock()  # serializes inference on this replica
        self._model = None
        self._init_health(reload_after_errors)
        # 推理锁排队 EWMA (ms): 全实例 VAD 串行化的饱和信号, 准入控制探针
        self._lock_wait_ewma_ms = 0.0
        self._init_model()

    @property
    def _log_tag(self) -> str:
        return f"FSMNVADEngine#{self.name}" if self.name else "FSMNVADEngine"

    @property
    def lock_wait_ewma_ms(self) -> float:
        with self._state_lock:
            return self._lock_wait_ewma_ms

    def _init_model(self):
        try:
            from funasr import AutoModel

            logger.info(
                f"[{self._log_tag}] Loading VAD model '{self.config.model_name_or_path}' on {self.config.device}..."
            )
            self._model = AutoModel(
                model=self.config.model_name_or_path,
                model_revision=self.config.model_revision,
                device=self.config.device,
                max_end_silence_time=self.config.max_end_silence_time,
                speech_noise_thresh=self.config.speech_noise_thresh,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
            )
            logger.info(f"[{self._log_tag}] VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"[{self._log_tag}] Failed to load FunASR VAD model: {e}")
            raise

    def process_chunk(
        self, audio_bytes: bytes, cache: Dict[str, Any], is_final: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Process a single streaming chunk with FSMN-VAD.
        Returns: list of (start_ms, end_ms) tuples, milliseconds from the
        start of this cache generation (FunASR ``init_cache`` on empty cache
        restarts the clock at 0). Not ring-buffer absolute time.
        """
        if not audio_bytes and not is_final:
            return []
        if audio_bytes and len(audio_bytes) & 1:
            # 半个采样点: 截齐防 frombuffer ValueError (网关已挡一道, 此处兜底)
            audio_bytes = audio_bytes[:-1]

        reload_reason: Optional[str] = None
        t_lock0 = time.perf_counter()
        with self._lock:
            lock_wait_ms = (time.perf_counter() - t_lock0) * 1000.0
            model = self._model
            if model is None:
                # Model is unloaded and reloading in the background: report no
                # boundaries for this chunk instead of blocking the executor.
                return []

            # Prepare kwargs for streaming VAD.
            # max_end_silence_time must be passed per-call: funasr only applies a
            # fixed threshold when it is present in generate kwargs, otherwise it
            # falls back to a dynamic silence schedule designed for non-streaming
            # input (thresholds of 1-2s that prevent streaming segmentation).
            kwargs = {
                "cache": cache,
                "is_final": is_final,
                "chunk_size": 200,  # 200ms default chunk for FSMN VAD
                "max_end_silence_time": self.config.max_end_silence_time,
                "speech_noise_thres": self.config.speech_noise_thresh,
            }

            # Feed float32 samples, never raw bytes: funasr's load_bytes() sniffs
            # container magic on bytes input and can misdetect raw PCM16 frames as
            # MP3 (near-silence dither patterns), dropping or garbling the chunk.
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                res = model.generate(input=samples, **kwargs)
            except Exception as e:
                logger.error(f"[{self._log_tag}] Error in VAD generate: {e}")
                reload_reason = self._note_error(e)
                segments: List[Tuple[int, int]] = []
            else:
                self._note_success()
                segments = []
                for seg in (res[0].get("value", []) if res else []):
                    if len(seg) >= 2:
                        segments.append((int(seg[0]), int(seg[1])))

        # Trigger the reload only after releasing the inference lock
        with self._state_lock:
            self._lock_wait_ewma_ms = 0.9 * self._lock_wait_ewma_ms + 0.1 * lock_wait_ms
        metrics.observe("asr_vad_lock_wait_ms", lock_wait_ms)
        if reload_reason is not None:
            self._start_reload(reload_reason)
        return segments
