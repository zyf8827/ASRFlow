import threading
import time
from typing import Optional

from loguru import logger

from core.metrics.prometheus_metrics import metrics


class SelfHealingMixin:
    """
    Unload-and-reload self-healing for engines that wrap ONE model object in
    `self._model` behind an inference lock `self._lock` (ONNX batched
    streaming engine, FSMN-VAD engine, ...).

    Host contract:
      - call `_init_health(reload_after_errors)` in `__init__` (sets state
        lock, counters; does not touch `_model`/`_lock`)
      - `self._model`: the loaded model, or None while a reload is in flight;
        hosts must read it under `self._lock` and fail soft when None
      - `self._init_model()`: (re)load the model into `self._model`; raise on
        failure (retried with capped exponential backoff, forever)
      - `self._log_tag`: log prefix (property or attribute)
      - on an inference exception, INSIDE the inference lock:
        `reason = self._note_error(err)`, then return the engine's empty result
      - on success, inside the inference lock: `self._note_success()`
      - AFTER releasing the inference lock: `if reason: self._start_reload(reason)`

    Lock ordering: the only legal nesting is `_lock` -> `_state_lock`;
    `_start_reload` takes the two locks in separate, non-nested blocks.
    Metric names are class attributes so each engine family reports its own
    series (streaming vs VAD).
    """

    METRIC_ERRORS = "asr_streaming_engine_errors"
    METRIC_RELOADS = "asr_streaming_engine_reloads"
    METRIC_RELOADING_GAUGE = "asr_streaming_engine_reloading"

    RELOAD_BACKOFF_START_SEC = 1.0
    RELOAD_BACKOFF_MAX_SEC = 60.0

    def _init_health(self, reload_after_errors: int = 3):
        self.reload_after_errors = max(1, reload_after_errors)
        self._state_lock = threading.Lock()  # guards health/reload bookkeeping
        self._consecutive_errors = 0
        self._reloading = False
        self._reload_generation = 0

    def is_available(self) -> bool:
        """Health hint: model loaded and not reloading."""
        return self._model is not None and not self._reloading

    def _note_success(self):
        with self._state_lock:
            self._consecutive_errors = 0

    def _note_error(self, err: Exception) -> Optional[str]:
        """Count an inference failure; return a reload reason once the
        consecutive-failure threshold is reached."""
        with self._state_lock:
            self._consecutive_errors += 1
            consecutive = self._consecutive_errors
        metrics.inc_counter(self.METRIC_ERRORS)
        if consecutive >= self.reload_after_errors:
            return f"{consecutive} consecutive inference failures (last: {err})"
        return None

    def _start_reload(self, reason: str):
        """Unload the broken model and reload it on a background thread.

        Idempotent: if a reload is already in flight the call returns
        immediately, so concurrent erroring callers cannot spawn several
        reload threads.
        """
        with self._state_lock:
            if self._reloading:
                return
            self._reloading = True

        with self._lock:
            # Drop the reference first: queued callers fast-fail while the
            # reload runs, and the broken model's memory is freed before the
            # replacement loads (peak stays at ~1 model).
            self._model = None

        metrics.inc_counter(self.METRIC_RELOADS)
        metrics.inc_gauge(self.METRIC_RELOADING_GAUGE)
        logger.error(f"[{self._log_tag}] Unloading model for reload: {reason}")
        threading.Thread(
            target=self._reload_worker,
            name=f"{self.__class__.__name__.lower()}-reload-{getattr(self, 'name', None) or '0'}",
            daemon=True,
        ).start()

    def _reload_worker(self):
        backoff = self.RELOAD_BACKOFF_START_SEC
        while True:
            try:
                self._init_model()
            except Exception as e:
                logger.error(
                    f"[{self._log_tag}] Model reload failed: {e}; "
                    f"retrying in {backoff:.0f}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.RELOAD_BACKOFF_MAX_SEC)
                continue

            with self._state_lock:
                self._consecutive_errors = 0
                self._reloading = False
                self._reload_generation += 1
                generation = self._reload_generation
            metrics.dec_gauge(self.METRIC_RELOADING_GAUGE)
            logger.info(
                f"[{self._log_tag}] Model reloaded successfully (generation {generation}); "
                "sessions resume automatically."
            )
            return
