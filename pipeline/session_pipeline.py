import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from loguru import logger

from core.session import ClientSession, SentenceRecord
from core.vad.base import BaseVADEngine
from core.streaming_asr.base import BaseStreamingASREngine
from core.final_asr.qwen_queue import FinalQueue
from core.speaker.base import BaseSpeakerEngine
from core.speaker.recluster import global_recluster_speakers
from core.audio_energy import pcm16_rms_dbfs
from core.guard.consistency_guard import ConsistencyGuard, check_ngram_repetition
from core.itn.itn_processor import ITNProcessor
from core.hotword.hotword_manager import HotwordManager
from core.punc.base import BasePuncEngine
from core.metrics.prometheus_metrics import metrics
from config.settings import AppConfig


class SessionPipeline:
    """
    Heterogeneous 2-Pass Real-time ASR Pipeline for a single WebSocket session.
    Orchestrates:
      - Audio Ring Buffer & Pre-roll
      - Streaming FSMN-VAD (sentence boundaries)
      - Streaming Paraformer (Pass-1 real-time partials & provisional)
      - Final Queue Micro-Batching with Qwen3-ASR (Pass-2 high precision)
      - ERes2Net Speaker Diarization (real-time tracking + session-end re-clustering)
      - Consistency & Hallucination Guard
      - ITN & Hotword Management
    """

    def __init__(
        self,
        session: ClientSession,
        vad_engine: BaseVADEngine,
        streaming_asr_engine: BaseStreamingASREngine,
        final_queue: FinalQueue,
        speaker_engine: BaseSpeakerEngine,
        punc_engine: Optional[BasePuncEngine],
        guard: ConsistencyGuard,
        itn: ITNProcessor,
        hotword_mgr: HotwordManager,
        thread_pool: ThreadPoolExecutor,
        output_queue: asyncio.Queue,
        config: AppConfig,
    ):
        self.session = session
        self.vad_engine = vad_engine
        self.streaming_asr_engine = streaming_asr_engine
        self.final_queue = final_queue
        self.speaker_engine = speaker_engine
        self.punc_engine = punc_engine
        self.guard = guard
        self.itn = itn
        self.hotword_mgr = hotword_mgr
        self.thread_pool = thread_pool
        self.output_queue = output_queue
        self.config = config

        # Speaker tracker is owned by the session so clusters survive a resume
        self.speaker_tracker = session.speaker_tracker

        self._hotwords = self.hotword_mgr.format_paraformer_hotwords(session.hotwords)
        self._current_segment_start_ms: int = -1

    def _reset_vad_stream(self, origin_ms: int) -> None:
        """Drop VAD cache and rebase its clock onto the ring-buffer timeline.

        FunASR and Mock both restart timestamps at 0 on an empty cache.
        Stashing the origin on the session (not in the cache dict) is
        required: FunASR only calls ``init_cache`` when ``len(cache)==0``.
        """
        self.session.vad_cache = {}
        self.session.vad_timeline_origin_ms = max(0, int(origin_ms))

    def _vad_to_ring_ms(self, vad_ms: int) -> int:
        """Map a cache-relative VAD timestamp onto ring-buffer absolute ms."""
        if vad_ms < 0:
            return vad_ms
        return vad_ms + self.session.vad_timeline_origin_ms

    def _final_history_context(self, current_sentence_id: int) -> str:
        """S3: tail of already-finalized second-pass text; empty if none yet.

        Does not wait for in-flight finals (that would add latency). This
        sentence's first-pass provisional is intentionally not used: it
        copies Paraformer errors into Qwen.
        """
        limit = self.config.final_asr.context_history_chars
        if limit <= 0:
            return ""
        parts = []
        for rec in sorted(self.session.sentences.values(), key=lambda r: r.sentence_id):
            if rec.sentence_id >= current_sentence_id:
                continue
            if rec.is_committed and rec.final_text:
                parts.append(rec.final_text)
        if not parts:
            return ""
        return "".join(parts)[-limit:]

    def update_hotwords(self, hotwords: Optional[List[str]]):
        """Apply a mid-session hotword update to both passes."""
        self.session.hotwords = hotwords or []
        self._hotwords = self.hotword_mgr.format_paraformer_hotwords(self.session.hotwords)

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        Main entry point for incoming PCM16 audio frames from client.
        """
        if not audio_bytes or not self.session.is_active or self.session.is_stopping:
            return

        self.session.touch()
        loop = asyncio.get_running_loop()

        # 1. Write raw audio into ring buffer (updates timeline)
        current_time_ms = self.session.ring_buffer.write(audio_bytes)

        voice_cfg = self.config.voice
        was_in_speech = self.session.is_in_speech

        # L0: near-zero digital silence, not currently in a VAD segment.
        # RMS is only computed out of speech (short-circuit): in-speech chunks
        # never use it, and pure-Python RMS costs ~1.25ms/s if run always-on.
        near_silence = (
            not was_in_speech
            and pcm16_rms_dbfs(audio_bytes) < voice_cfg.silence_floor_dbfs
        )
        skip_infer = near_silence and voice_cfg.skip_inference_on_silence

        # 2. Parallel execution of VAD and Streaming ASR in ThreadPool
        t0 = time.time()
        vad_task = loop.run_in_executor(
            self.thread_pool,
            self.vad_engine.process_chunk,
            audio_bytes,
            self.session.vad_cache,
            False,
        )

        if skip_infer:
            vad_segments = await vad_task
            partial_text = ""
            metrics.inc_counter("asr_silence_dropped_chunks_total")
        else:
            streaming_task = loop.run_in_executor(
                self.thread_pool,
                self.streaming_asr_engine.process_chunk,
                audio_bytes,
                self.session.streaming_cache,
                False,
                self._hotwords,
            )
            vad_segments, partial_text = await asyncio.gather(vad_task, streaming_task)
            if near_silence and partial_text:
                partial_text = ""
                metrics.inc_counter("asr_silence_dropped_chunks_total")

        infer_latency_ms = (time.time() - t0) * 1000.0
        metrics.observe("asr_paraformer_latency_ms", infer_latency_ms)
        metrics.observe("asr_vad_streaming_latency_ms", infer_latency_ms)

        # Accumulate first so a VAD endpoint on this chunk still includes
        # the newly confirmed tokens. Emission is gated after VAD events
        # so a start on this chunk can flush pre-start tokens immediately.
        had_new_tokens = bool(partial_text)
        if had_new_tokens:
            self.session.current_partial_text += partial_text
        metrics.observe("asr_partial_text_len", len(self.session.current_partial_text))

        # 3. VAD sentence boundaries BEFORE partial emit (L1 ordering)
        await self._handle_vad_segments(vad_segments)

        # 4. Emit Partial Result if updated and (if gated) in speech
        if had_new_tokens:
            await self._maybe_emit_partial(current_time_ms)

        # Force-cut: continuous speech without any VAD pause (e.g. fluent
        # reading) is split once the configured max single-sentence duration
        # is reached, bounding segment length for the second pass
        max_speech_ms = self.config.vad.max_speech_duration_ms
        if (
            max_speech_ms > 0
            and self.session.is_in_speech
            and self.session.speech_start_ms != -1
            and current_time_ms - self.session.speech_start_ms >= max_speech_ms
        ):
            logger.info(
                f"[{self.session.session_id}] Max speech duration ({max_speech_ms}ms) reached, "
                f"force-cutting sentence [{self.session.speech_start_ms}~{current_time_ms}ms]"
            )
            # Force-cut keeps speech active: adjust gauge to avoid leak (dec then inc)
            metrics.dec_gauge("asr_active_speech_sessions")
            await self._handle_sentence_endpoint(self.session.speech_start_ms, current_time_ms)
            metrics.inc_gauge("asr_active_speech_sessions")
            # Speech continues after the cut: re-arm the start so the next
            # natural VAD endpoint closes the follow-up sentence
            self.session.speech_start_ms = current_time_ms
            self._current_segment_start_ms = current_time_ms
            # Update watermark
            self.session.last_committed_end_ms = current_time_ms

        # 5. L2 watchdog: non-speech accumulation cap + energy pre-check
        await self._maybe_watchdog(current_time_ms)

    def _resolve_partial_start_ms(self, current_time_ms: int) -> int:
        """Stable start_ms: VAD segment start, else first latched pre-start value."""
        if self._current_segment_start_ms != -1:
            return self._current_segment_start_ms
        if self.session.pending_partial_start_ms < 0:
            self.session.pending_partial_start_ms = max(0, current_time_ms - 2000)
        return self.session.pending_partial_start_ms

    def _reset_partial_start_latch(self) -> None:
        self.session.pending_partial_start_ms = -1

    def _note_outgoing_text(self, text: str, kind: str) -> None:
        """Output-queue ngram-loop canary (alerts on degenerate rows escaping L0-L2)."""
        if not text:
            return
        if check_ngram_repetition(
            text,
            ngram=self.config.guard.repetition_ngram,
            max_repeat=self.config.guard.repetition_max_count,
        ):
            metrics.inc_counter("asr_degenerate_ngram_detected_total")
            logger.warning(
                f"[{self.session.session_id}] ngram loop detected in {kind} "
                f"(len={len(text)})"
            )

    async def _maybe_emit_partial(self, current_time_ms: int) -> None:
        if not self.session.current_partial_text:
            return
        start_ms = self._resolve_partial_start_ms(current_time_ms)
        if self.config.voice.suppress_prestart_partials and not self.session.is_in_speech:
            metrics.inc_counter("asr_partial_suppressed_total")
            return
        seq = self.session.next_partial_seq()
        text = self.session.current_partial_text
        self._note_outgoing_text(text, "partial")
        await self.output_queue.put(
            {
                "type": "partial",
                "mode": "2pass-online",
                "session_id": self.session.session_id,
                "seq": seq,
                "start_ms": start_ms,
                "end_ms": current_time_ms,
                "begin_time": start_ms,
                "end_time": current_time_ms,
                "text": text,
                "wav_name": "",
                "is_final": False,
            }
        )

    async def _handle_vad_segments(self, vad_segments) -> None:
        for raw_start, raw_end in vad_segments:
            seg_start = self._vad_to_ring_ms(raw_start)
            seg_end = self._vad_to_ring_ms(raw_end)
            if seg_start != -1:
                # Speech Start Detected
                self.session.is_in_speech = True
                self.session.speech_start_ms = seg_start
                self._current_segment_start_ms = seg_start
                self._reset_partial_start_latch()
                metrics.inc_gauge("asr_active_speech_sessions")
                logger.debug(
                    f"[{self.session.session_id}] VAD Speech Start at {seg_start}ms"
                )

            if seg_end != -1:
                # Speech End Detected (Endpoint)
                metrics.dec_gauge("asr_active_speech_sessions")
                self.session.is_in_speech = False
                actual_start_ms = self.session.speech_start_ms if self.session.speech_start_ms != -1 else max(0, seg_end - 3000)
                # Clamp to watermark to avoid overlap with prior COMMIT
                actual_start_ms = max(actual_start_ms, self.session.last_committed_end_ms)
                actual_end_ms = max(seg_end, actual_start_ms)
                self.session.speech_start_ms = -1
                self._current_segment_start_ms = -1
                self._reset_partial_start_latch()

                logger.debug(
                    f"[{self.session.session_id}] VAD Speech End at {seg_end}ms (duration={actual_end_ms - actual_start_ms}ms)"
                )

                await self._handle_sentence_endpoint(actual_start_ms, actual_end_ms)

    def _discard_unsent_partial(self, now_ms: int) -> None:
        """Local drop of gated garbage: backend never saw it, so no empty final."""
        self.session.current_partial_text = ""
        self.session.streaming_cache = {}
        self.session.last_committed_end_ms = max(self.session.last_committed_end_ms, now_ms)
        self._current_segment_start_ms = -1
        self._reset_partial_start_latch()

    def _watchdog_over_rate_limit(self, now_ts: float) -> bool:
        hour_ago = now_ts - 3600.0
        times = self.session.watchdog_force_times
        while times and times[0] < hour_ago:
            times.popleft()
        return len(times) >= self.config.voice.watchdog_rate_limit_per_hour

    async def _maybe_watchdog(self, current_time_ms: int) -> None:
        """L2: bounded adjudication of gated non-speech accumulation.

        Real VAD-missed speech (~3-5 chars/s) and noise hallucination
        (~0.3 chars/s) both accumulate while unsent; on reaching the char
        threshold we force ONE Qwen+Guard adjudication of the trailing
        segment instead of guessing locally — a local guess can only either
        burn Qwen calls or drop testimony. Pathological noise sessions
        (~2.4% of tasks) are bounded by the hourly rate limit; Guard drops
        pure-noise segments via the empty-Qwen path.
        """
        cfg = self.config.voice
        if not cfg.watchdog_enable:
            return
        if self.session.is_in_speech:
            return
        if len(self.session.current_partial_text) < cfg.watchdog_min_chars:
            return

        now_ts = time.time()
        if self._watchdog_over_rate_limit(now_ts):
            metrics.inc_counter("asr_watchdog_force_total", labels={"result": "ratelimit"})
            logger.info(
                f"[{self.session.session_id}] Watchdog rate-limited, discarding "
                f"{len(self.session.current_partial_text)} chars"
            )
            self._discard_unsent_partial(current_time_ms)
            return

        self.session.watchdog_force_times.append(now_ts)
        metrics.inc_counter("asr_watchdog_force_total", labels={"result": "force"})
        logger.info(
            f"[{self.session.session_id}] Watchdog force-cut "
            f"(chars={len(self.session.current_partial_text)}) at {current_time_ms}ms"
        )
        start_ms = max(
            self.session.last_committed_end_ms,
            current_time_ms - cfg.watchdog_max_segment_ms,
        )
        await self._handle_sentence_endpoint(start_ms, current_time_ms)
        if self.session.current_partial_text:
            self._discard_unsent_partial(current_time_ms)

    async def _handle_sentence_endpoint(self, start_ms: int, end_ms: int):
        """
        Handles VAD Endpoint by triggering:
          - Path A: Immediate Paraformer flush for Provisional result.
          - Path B: Background Qwen3-ASR Final transcription + Speaker Diarization.
        """
        # Unified watermark: every caller (VAD / COMMIT / STOP / force-cut)
        # must not start before the last finalized end. Skip empty/overlap
        # intervals so a rewound VAD clock cannot emit duplicate or empty finals.
        start_ms = max(start_ms, self.session.last_committed_end_ms)
        if end_ms <= start_ms:
            logger.debug(
                f"[{self.session.session_id}] Skip empty/overlap sentence "
                f"[{start_ms}~{end_ms}ms] watermark={self.session.last_committed_end_ms}ms"
            )
            return
        self.session.last_committed_end_ms = end_ms
        self._reset_partial_start_latch()
        loop = asyncio.get_running_loop()

        # Path A: Flush Streaming Paraformer to produce Provisional text
        provisional_raw = await loop.run_in_executor(
            self.thread_pool,
            self.streaming_asr_engine.flush,
            self.session.streaming_cache,
            self._hotwords,
        )

        # Combine the accumulated partial with the flush tail (both are
        # incremental) to form the full provisional sentence
        provisional_text = (self.session.current_partial_text + provisional_raw).strip()
        self.session.current_partial_text = ""
        # 换新引用而非原地 clear: 旧 dict 可能仍被在途读者持有 (batcher 队列
        # 中的旧请求、断连后的孤儿 chunk), 原地掏空会使其 KeyError 崩溃
        self.session.streaming_cache = {}

        # Apply ITN on provisional text
        provisional_text = self.itn.normalize(provisional_text)

        # Apply optional real-time punc
        if self.config.punc.enable_realtime and self.punc_engine and provisional_text:
            provisional_text = await loop.run_in_executor(
                self.thread_pool,
                self.punc_engine.add_punctuation,
                provisional_text,
            )

        sentence_id = self.session.next_sentence_id()
        record = SentenceRecord(
            sentence_id=sentence_id,
            start_ms=start_ms,
            end_ms=end_ms,
            provisional_text=provisional_text,
        )
        self.session.sentences[sentence_id] = record

        # Push Provisional text to client immediately (low-latency first view)
        self._note_outgoing_text(provisional_text, "provisional")
        await self.output_queue.put(
            {
                "type": "provisional",
                "mode": "2pass-provisional",
                "session_id": self.session.session_id,
                "sentence_id": sentence_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "begin_time": start_ms,
                "end_time": end_ms,
                "text": provisional_text,
                "wav_name": "",
                "is_final": False,
            }
        )

        # Path B: Extract audio segment from RingBuffer.
        # Speaker embedding keeps the full pre-roll (more voice samples);
        # Qwen gets a short lead-in only — the inter-sentence silence is at
        # least max_end_silence_time, so a small pre-roll is silence-only and
        # avoids re-transcribing the previous sentence's tail into this one.
        full_segment_pcm = self.session.ring_buffer.get_segment_with_preroll(
            start_ms=start_ms,
            end_ms=end_ms,
            custom_preroll_ms=min(self.config.audio.pre_roll_ms, 200),
        )
        speaker_segment_pcm = self.session.ring_buffer.get_segment_with_preroll(
            start_ms=start_ms,
            end_ms=end_ms,
            custom_preroll_ms=self.config.audio.pre_roll_ms,
        )

        # Launch Final ASR and Speaker task in background
        task = asyncio.create_task(
            self._process_final_segment(
                sentence_id=sentence_id,
                start_ms=start_ms,
                end_ms=end_ms,
                audio_bytes=full_segment_pcm,
                speaker_audio_bytes=speaker_segment_pcm,
                provisional_text=provisional_text,
            )
        )
        self.session.register_pending_task(task)

    async def _process_final_segment(
        self,
        sentence_id: int,
        start_ms: int,
        end_ms: int,
        audio_bytes: bytes,
        provisional_text: str,
        speaker_audio_bytes: Optional[bytes] = None,
    ):
        """
        Background task processing Qwen3-ASR Final and ERes2Net Speaker Diarization in parallel.
        """
        duration_ms = max(100, end_ms - start_ms)
        loop = asyncio.get_running_loop()

        # 1. Start Speaker Embedding in ThreadPool
        spk_task = None
        if self.session.enable_spk:
            spk_task = loop.run_in_executor(
                self.thread_pool,
                self.speaker_engine.extract_embedding,
                speaker_audio_bytes if speaker_audio_bytes is not None else audio_bytes,
            )

        # 2. Submit to Qwen Final Queue
        qwen_text = ""
        final_source = "qwen3-asr"
        revision_distance = 0.0
        needs_review = False
        selected_text = provisional_text

        try:
            # S3: already-finalized history tail. Do not feed this sentence's
            # first-pass provisional as context (S2 pollutes Qwen).
            history_context = self._final_history_context(sentence_id)
            qwen_raw = await self.final_queue.submit(
                audio_bytes=audio_bytes,
                context=history_context or None,
                hotwords=self.session.hotwords,
                timeout_sec=self.config.final_asr.hard_timeout_sec,
            )
            qwen_text = qwen_raw.strip()

            # 3. Consistency Guard check (executor: Levenshtein DP is pure
            # Python and would otherwise stall the event loop for all sessions)
            decision = await loop.run_in_executor(
                self.thread_pool,
                self.guard.evaluate,
                provisional_text,
                qwen_text,
                duration_ms,
            )

            selected_text = decision.selected_text
            final_source = decision.final_source
            revision_distance = decision.revision_distance
            needs_review = decision.needs_review

            if decision.rejection_reason:
                logger.info(
                    f"[{self.session.session_id}] Sentence {sentence_id} Guard: {decision.rejection_reason}"
                )

        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.session.session_id}] Sentence {sentence_id} Qwen timed out, fallback to provisional."
            )
            final_source = "paraformer-fallback"
            selected_text = provisional_text
            revision_distance = 1.0
            needs_review = False
        except Exception as e:
            logger.error(
                f"[{self.session.session_id}] Sentence {sentence_id} Qwen inference failed: {e}, fallback."
            )
            final_source = "paraformer-fallback"
            selected_text = provisional_text
            revision_distance = 1.0
            needs_review = False

        # 4. ITN & Hotword post-processing
        normalized_text = self.itn.normalize(selected_text)
        final_text = self.hotword_mgr.postprocess_replace(
            normalized_text, custom_dict=self.session.postprocess_hotwords or None
        )

        # 5. Await Speaker Embedding and Classify
        speaker_id = "SPK1"
        embedding = None
        if spk_task:
            try:
                embedding = await spk_task
                if embedding is not None:
                    spk = self.speaker_tracker.classify_and_update(embedding)
                    if spk:
                        speaker_id = spk
            except Exception as e:
                logger.error(f"[{self.session.session_id}] Speaker classification error: {e}")

        # 6. Update Record in Session
        record = self.session.sentences.get(sentence_id)
        if record:
            record.final_text = final_text
            record.speaker = speaker_id
            record.final_source = final_source
            record.revision_distance = revision_distance
            record.needs_review = needs_review
            record.is_committed = True
            record.speaker_embedding = embedding
            record.committed_at = time.time()

        # Update Metrics
        metrics.inc_counter("asr_sentences_finalized")
        metrics.observe("asr_revision_distance", revision_distance)
        if needs_review:
            metrics.inc_counter("asr_sentences_needs_review")

        # 7. Push Final Result to Client
        self._note_outgoing_text(final_text, "final")
        await self.output_queue.put(
            {
                "type": "final",
                "mode": "2pass-offline",
                "session_id": self.session.session_id,
                "sentence_id": sentence_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "begin_time": start_ms,
                "end_time": end_ms,
                "speaker": speaker_id,
                "text": final_text,
                "final_source": final_source,
                "revision_distance": round(revision_distance, 4),
                "needs_review": needs_review,
                "wav_name": "",
                "is_final": True,
            }
        )

        logger.info(
            f"[{self.session.session_id}] Sentence {sentence_id} Final: "
            f"[{speaker_id}] '{final_text}' (src={final_source}, dist={revision_distance:.2f})"
        )

    async def process_commit(self):
        """
        Handles manual client-driven COMMIT message:
        Immediately finalizes the current utterance without waiting for VAD silence.
        """
        if not self.session.is_active or self.session.is_stopping:
            return

        current_time_ms = self.session.ring_buffer.total_ms

        # Silence COMMIT optimization: no accumulated text and not in speech
        # -> advance watermark locally, no second pass. Leftover gated text
        # (L1) is NOT dropped here: it falls through to the normal endpoint
        # below so Qwen+Guard adjudicates it once.
        if not self.session.is_in_speech and not self.session.current_partial_text:
            logger.info(f"[{self.session.session_id}] COMMIT in silence, emitting empty final locally")
            # Ensure watermark advances to avoid overlap
            self.session.last_committed_end_ms = max(self.session.last_committed_end_ms, current_time_ms)
            self._current_segment_start_ms = -1
            self._reset_partial_start_latch()
            # Reset VAD stream so the next utterance is not on a rewound clock
            self._reset_vad_stream(current_time_ms)
            return

        if self.session.speech_start_ms != -1:
            start_ms = self.session.speech_start_ms
        elif self.session.pending_partial_start_ms >= 0:
            start_ms = self.session.pending_partial_start_ms
        else:
            start_ms = max(0, current_time_ms - 2000)
        start_ms = max(start_ms, self.session.last_committed_end_ms)
        end_ms = current_time_ms

        logger.info(
            f"[{self.session.session_id}] Client-driven COMMIT triggered for [{start_ms}~{end_ms}ms]"
        )

        if self.session.is_in_speech:
            metrics.dec_gauge("asr_active_speech_sessions")
        self.session.is_in_speech = False
        self.session.speech_start_ms = -1
        self._current_segment_start_ms = -1
        self._reset_partial_start_latch()
        # Reset VAD cache to avoid stale in_speech producing overlapping endpoint
        # (换新引用而非原地 clear, 理由同 _handle_sentence_endpoint).
        # Rebase the VAD clock onto the ring-buffer time at COMMIT: FunASR/Mock
        # restart timestamps at 0 on an empty cache.
        self._reset_vad_stream(end_ms)

        # 零长段: 强制切句重臂 speech_start 后紧跟 COMMIT (无任何新音频) 会切出
        # [t~t] 空段, 无音频可收尾; 与静默 COMMIT 同语义, 只推进水位不调二遍
        if end_ms <= start_ms:
            self.session.last_committed_end_ms = max(self.session.last_committed_end_ms, end_ms)
            return

        await self._handle_sentence_endpoint(start_ms, end_ms)

    async def process_stop(self):
        """
        Handles STOP message from client:
          1. Finalizes VAD & flushes any remaining speech.
          2. Waits for all pending Qwen Final tasks.
          3. Runs global speaker re-clustering.
          4. Emits session_finished event with full transcript.
        """
        if self.session.is_stopping:
            return
        self.session.is_stopping = True

        logger.info(f"[{self.session.session_id}] Processing STOP request...")
        loop = asyncio.get_running_loop()

        # 1. Finalize VAD
        current_time_ms = self.session.ring_buffer.total_ms
        vad_segments = await loop.run_in_executor(
            self.thread_pool,
            self.vad_engine.process_chunk,
            b"",
            self.session.vad_cache,
            True,
        )

        for _raw_start, raw_end in vad_segments:
            seg_end = self._vad_to_ring_ms(raw_end)
            if seg_end != -1:
                start_ms = self.session.speech_start_ms if self.session.speech_start_ms != -1 else max(0, seg_end - 2000)
                await self._handle_sentence_endpoint(start_ms, seg_end)

        # If speech was still open, close it
        if self.session.is_in_speech:
            start_ms = self.session.speech_start_ms if self.session.speech_start_ms != -1 else max(0, current_time_ms - 2000)
            # 零长段防御: 强制切句重臂后立即 stop 会得到 [t~t], 无音频可收尾
            if current_time_ms > start_ms:
                await self._handle_sentence_endpoint(start_ms, current_time_ms)
            self.session.is_in_speech = False
        elif self.config.voice.watchdog_enable and self.session.current_partial_text:
            # L2: leftover gated text at STOP — last chance to recover
            # VAD-missed speech; Qwen+Guard adjudicates (no local guess,
            # Guard drops pure noise via the empty-Qwen path)
            start_ms = max(
                self.session.last_committed_end_ms,
                current_time_ms - self.config.voice.watchdog_max_segment_ms,
            )
            if current_time_ms > start_ms:
                await self._handle_sentence_endpoint(start_ms, current_time_ms)

        # 2. Wait for pending Qwen final jobs
        await self.session.wait_for_pending_tasks(timeout=self.config.final_asr.hard_timeout_sec + 1.0)

        # 3. Session-end Global Speaker Re-Clustering (executor: O(n^2)
        # clustering on the loop would stall every session for long meetings)
        if self.session.enable_spk and self.session.sentences:
            try:
                sentence_list = [rec.to_dict() for rec in self.session.sentences.values()]
                id_map = {d["sentence_id"]: d for d in sentence_list}
                for s_id, rec in self.session.sentences.items():
                    if s_id in id_map:
                        id_map[s_id]["speaker_embedding"] = rec.speaker_embedding

                global_speaker_map = await loop.run_in_executor(
                    None,  # 默认执行器: 一次性 O(n^2) 任务, 不占推理线程池
                    global_recluster_speakers,
                    sentence_list,
                    self.session.expected_speakers,
                )

                for s_id, refined_spk in global_speaker_map.items():
                    if s_id in self.session.sentences:
                        self.session.sentences[s_id].speaker = refined_spk
            except Exception as e:
                logger.error(f"[{self.session.session_id}] Global re-clustering failed: {e}")

        # 4. Compile full transcript
        full_transcript = [
            rec.to_dict()
            for rec in sorted(self.session.sentences.values(), key=lambda r: r.sentence_id)
        ]

        # 5. Emit session_finished message
        await self.output_queue.put(
            {
                "type": "session_finished",
                "mode": "2pass-offline",
                "event": "stopped",
                "session_id": self.session.session_id,
                "total_sentences": len(self.session.sentences),
                "duration_ms": current_time_ms,
                "sentences": full_transcript,
                "transcript": full_transcript,
                "is_final": True,
            }
        )

        logger.info(
            f"[{self.session.session_id}] Session finished with {len(self.session.sentences)} sentences."
        )
