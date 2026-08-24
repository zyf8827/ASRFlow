import time
import uuid
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from core.ring_buffer import AudioRingBuffer
from core.speaker.speaker_tracker import IncrementalSpeakerTracker
from config.settings import AudioConfig, AppConfig


@dataclass
class SentenceRecord:
    sentence_id: int
    start_ms: int
    end_ms: int
    provisional_text: str
    final_text: str = ""
    speaker: Optional[str] = None
    final_source: str = "qwen3-asr"  # "qwen3-asr" | "paraformer-fallback"
    revision_distance: float = 0.0
    needs_review: bool = False
    is_committed: bool = False
    speaker_embedding: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    committed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "text": self.final_text if self.final_text else self.provisional_text,
            "provisional_text": self.provisional_text,
            "final_text": self.final_text,
            "final_source": self.final_source,
            "revision_distance": round(self.revision_distance, 4),
            "needs_review": self.needs_review,
            "is_committed": self.is_committed,
            "is_final": bool(self.is_committed and self.final_text is not None),
        }


class ClientSession:
    """
    State container for a single real-time WebSocket session.
    Manages session-specific model caches, audio ring buffer, speaker clustering,
    timeline synchronization, and pending final ASR tasks.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        language: str = "zh",
        enable_spk: bool = True,
        expected_speakers: int = 2,
        hotwords: Optional[List[str]] = None,
        app_config: Optional[AppConfig] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.language = language
        self.enable_spk = enable_spk
        self.expected_speakers = expected_speakers
        self.hotwords = hotwords or []
        self.config = app_config or AppConfig()

        self.created_at = time.time()
        self.last_active_at = time.time()

        # Audio Timeline & Ring Buffer
        audio_cfg = self.config.audio
        self.ring_buffer = AudioRingBuffer(
            sample_rate=audio_cfg.sample_rate,
            channels=audio_cfg.channels,
            bytes_per_sample=audio_cfg.bytes_per_sample,
            max_duration_sec=audio_cfg.ring_buffer_duration_sec,
            pre_roll_ms=audio_cfg.pre_roll_ms,
        )

        # VAD State & Cache
        self.vad_cache: Dict[str, Any] = {}
        self.is_in_speech: bool = False
        self.speech_start_ms: int = -1
        # Watermark to avoid COMMIT/VAD overlap producing duplicate intervals
        self.last_committed_end_ms: int = 0
        # FunASR/Mock VAD timestamps restart at 0 whenever vad_cache is empty.
        # This origin maps those cache-relative ms onto the ring-buffer timeline.
        self.vad_timeline_origin_ms: int = 0

        # Streaming Paraformer State & Cache
        self.streaming_cache: Dict[str, Any] = {}
        self.current_partial_text: str = ""
        self.partial_seq: int = 0
        # Latched start_ms for pre-VAD partials (COMMIT / logs); reset on
        # VAD start/endpoint/COMMIT/watchdog discard.
        self.pending_partial_start_ms: int = -1

        # L2 watchdog rate-limit state (survives resume reattach)
        self.watchdog_force_times: deque = deque()

        # Sentence Tracking & Sequence
        self.sentence_seq: int = 0
        self.sentences: Dict[int, SentenceRecord] = {}

        # Per-session hotword post-processing dictionary ("wrong=>right")
        self.postprocess_hotwords: Dict[str, str] = {}

        # Concurrency & Task Management
        self.pending_final_tasks: Set[asyncio.Task] = set()
        self.is_active: bool = True
        self.is_stopping: bool = False

        # Detached (disconnect) state: set when the client drops and the
        # session is kept for a resume grace period
        self.detached_at: Optional[float] = None
        self.detached_pipeline: Optional[Any] = None

        # Speaker tracker lives on the session so cluster state survives a
        # resume reattach (pipeline instances are per-connection)
        self.speaker_tracker = IncrementalSpeakerTracker(
            similarity_threshold=self.config.speaker.similarity_threshold,
            expected_speakers=expected_speakers,
            max_speakers=self.config.speaker.max_speakers,
        )

    def touch(self):
        """Update last active timestamp."""
        self.last_active_at = time.time()

    def next_sentence_id(self) -> int:
        self.sentence_seq += 1
        return self.sentence_seq

    def next_partial_seq(self) -> int:
        self.partial_seq += 1
        return self.partial_seq

    def register_pending_task(self, task: asyncio.Task):
        self.pending_final_tasks.add(task)
        task.add_done_callback(self.pending_final_tasks.discard)

    async def wait_for_pending_tasks(self, timeout: float = 3.0):
        """Wait for all in-flight Qwen Final jobs to finish during STOP."""
        if not self.pending_final_tasks:
            return
        pending = list(self.pending_final_tasks)
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            from loguru import logger

            logger.warning(f"[{self.session_id}] wait_for_pending_tasks timed out, cancelling {len(pending)} tasks")
            for task in pending:
                if not task.done():
                    task.cancel()

    def commit_transcript(self) -> List[Dict[str, Any]]:
        """All finalized sentences in sentence_id order."""
        return [
            rec.to_dict()
            for rec in sorted(self.sentences.values(), key=lambda r: r.sentence_id)
        ]

    def cleanup(self):
        """Clean up memory and caches on session close."""
        self.is_active = False
        self.ring_buffer.clear()
        # 换新引用而非原地 clear: 同 ID 重建/收尾时, 旧会话的在途 executor
        # 线程与 batcher 旧请求可能仍持有旧 dict, 原地掏空会使其 KeyError
        self.vad_cache = {}
        self.vad_timeline_origin_ms = 0
        self.streaming_cache = {}
        for task in list(self.pending_final_tasks):
            if not task.done():
                task.cancel()
        self.pending_final_tasks.clear()
