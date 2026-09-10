"""L0 / L1 / L2 streaming hallucination guard (gateway only).

See docs/design_degenerate_rows_guard_2026-09-18.md. Backend nextDelta is
out of scope for this repo.
"""
import asyncio
import math
import struct
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from config.settings import AppConfig
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue
from core.guard.consistency_guard import ConsistencyGuard
from core.hotword.hotword_manager import HotwordManager
from core.itn.itn_processor import ITNProcessor
from core.session import ClientSession
from core.speaker.mock_speaker import MockSpeakerEngine
from core.vad.mock_vad import MockVADEngine
from pipeline.session_pipeline import SessionPipeline


def _pcm_tone(duration_ms: int, freq: float = 260.0, amp: int = 8000) -> bytes:
    n = duration_ms * 16
    samples = [int(amp * math.sin(2 * math.pi * freq * i / 16000.0)) for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


def _pcm_silence(duration_ms: int) -> bytes:
    return b"\x00\x00" * (duration_ms * 16)


def _drain(queue: asyncio.Queue):
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    return messages


class _SilentVAD:
    def process_chunk(self, audio_bytes, cache, is_final=False):
        cache["t"] = cache.get("t", 0) + (len(audio_bytes) // 32 if audio_bytes else 0)
        return []


class _TokenStreaming:
    """Emits a fixed token on every non-final chunk (hallucination stand-in)."""

    def __init__(self, token: str = "嗯"):
        self.token = token
        self.calls = 0

    def process_chunk(self, audio_bytes, cache, is_final=False, hotwords=None):
        self.calls += 1
        if is_final or not audio_bytes:
            return ""
        return self.token

    def flush(self, cache, hotwords=None):
        return ""


class _DelayedBurstStreaming:
    """Stay silent until ``start_after_ms``, then emit ``token`` once."""

    def __init__(self, start_after_ms: int = 3500, token: str = "漏检语音一二三四五"):
        self.start_after_ms = start_after_ms
        self.token = token

    def process_chunk(self, audio_bytes, cache, is_final=False, hotwords=None):
        if is_final or not audio_bytes:
            return ""
        acc = cache.get("ms", 0) + len(audio_bytes) // 32
        cache["ms"] = acc
        if cache.get("emitted"):
            return ""
        if acc >= self.start_after_ms:
            cache["emitted"] = True
            return self.token
        return ""

    def flush(self, cache, hotwords=None):
        return ""


def _base_config() -> AppConfig:
    cfg = AppConfig()
    cfg.streaming_asr.backend = "mock"
    cfg.vad.backend = "mock"
    cfg.final_asr.engine_type = "mock"
    cfg.speaker.backend = "mock"
    cfg.vad.max_speech_duration_ms = 60000
    cfg.voice.suppress_prestart_partials = True
    cfg.voice.watchdog_enable = False
    cfg.voice.skip_inference_on_silence = False
    return cfg


class TestL0L1Guard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = _base_config()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    async def asyncTearDown(self):
        self.thread_pool.shutdown(wait=False)

    def _pipeline(self, vad, streaming, session_id: str = "g") -> SessionPipeline:
        session = ClientSession(session_id=session_id, app_config=self.config)
        return SessionPipeline(
            session=session,
            vad_engine=vad,
            streaming_asr_engine=streaming,
            final_queue=None,
            speaker_engine=MockSpeakerEngine(self.config.speaker),
            punc_engine=None,
            guard=ConsistencyGuard(self.config.guard),
            itn=ITNProcessor(self.config.itn),
            hotword_mgr=HotwordManager(),
            thread_pool=self.thread_pool,
            output_queue=asyncio.Queue(),
            config=self.config,
        )

    async def _feed(self, pipeline: SessionPipeline, pcm: bytes, chunk_ms: int = 60):
        chunk_bytes = chunk_ms * 32
        for offset in range(0, len(pcm), chunk_bytes):
            await pipeline.process_audio_chunk(pcm[offset : offset + chunk_bytes])

    async def test_l0_drops_tokens_on_digital_silence(self):
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        await self._feed(pipeline, _pcm_silence(600))
        self.assertEqual(pipeline.session.current_partial_text, "")
        msgs = _drain(pipeline.output_queue)
        self.assertEqual([m for m in msgs if m.get("type") == "partial"], [])
        self.assertGreater(streaming.calls, 0)

    async def test_l0_skip_inference_does_not_call_engine(self):
        self.config.voice.skip_inference_on_silence = True
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        await self._feed(pipeline, _pcm_silence(300))
        self.assertEqual(streaming.calls, 0)
        self.assertEqual(pipeline.session.current_partial_text, "")

    async def test_l1_suppresses_prestart_partials_but_keeps_text(self):
        streaming = _TokenStreaming("你")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        await self._feed(pipeline, _pcm_tone(300, amp=20))
        self.assertEqual(pipeline.session.current_partial_text, "你" * 5)
        msgs = _drain(pipeline.output_queue)
        self.assertEqual([m for m in msgs if m.get("type") == "partial"], [])

    async def test_l1_first_in_speech_partial_includes_prestart_tokens(self):
        streaming = _TokenStreaming("你")
        vad = MockVADEngine(self.config.vad, energy_threshold=50.0)
        pipeline = self._pipeline(vad, streaming)
        await self._feed(pipeline, _pcm_tone(300, amp=20))
        self.assertFalse(pipeline.session.is_in_speech)
        self.assertEqual(len(pipeline.session.current_partial_text), 5)
        await self._feed(pipeline, _pcm_tone(240, amp=8000))
        self.assertTrue(pipeline.session.is_in_speech)
        partials = [m for m in _drain(pipeline.output_queue) if m.get("type") == "partial"]
        self.assertGreater(len(partials), 0)
        self.assertTrue(partials[0]["text"].startswith("你" * 5))

    async def test_start_ms_latches_when_l1_disabled(self):
        self.config.voice.suppress_prestart_partials = False
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        pipeline.session.ring_buffer.write(_pcm_silence(5000))
        await self._feed(pipeline, _pcm_tone(180, amp=20))
        partials = [m for m in _drain(pipeline.output_queue) if m.get("type") == "partial"]
        self.assertGreaterEqual(len(partials), 2)
        starts = {m["start_ms"] for m in partials}
        self.assertEqual(len(starts), 1)
        self.assertGreater(next(iter(starts)), 0)


class TestL2Watchdog(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = _base_config()
        self.config.voice.watchdog_enable = True
        self.config.voice.watchdog_min_chars = 8
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.final_queue = FinalQueue(
            MockQwenEngine(self.config.final_asr, simulated_latency_ms=1.0),
            self.config.final_asr,
        )
        self.final_queue.start()

    async def asyncTearDown(self):
        await self.final_queue.stop()
        self.thread_pool.shutdown(wait=False)

    def _pipeline(self, vad, streaming, session_id: str = "w") -> SessionPipeline:
        session = ClientSession(session_id=session_id, app_config=self.config)
        return SessionPipeline(
            session=session,
            vad_engine=vad,
            streaming_asr_engine=streaming,
            final_queue=self.final_queue,
            speaker_engine=MockSpeakerEngine(self.config.speaker),
            punc_engine=None,
            guard=ConsistencyGuard(self.config.guard),
            itn=ITNProcessor(self.config.itn),
            hotword_mgr=HotwordManager(),
            thread_pool=self.thread_pool,
            output_queue=asyncio.Queue(),
            config=self.config,
        )

    async def _feed(self, pipeline: SessionPipeline, pcm: bytes, chunk_ms: int = 60):
        chunk_bytes = chunk_ms * 32
        for offset in range(0, len(pcm), chunk_bytes):
            await pipeline.process_audio_chunk(pcm[offset : offset + chunk_bytes])

    async def test_watchdog_force_cuts_accumulation_to_guard(self):
        # 噪声幻觉攒满字数阈值 → 强切一段进 Qwen+Guard 裁决:
        # provisional 出句、累积清零、内存有界(不做本地能量判断)
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        await self._feed(pipeline, _pcm_tone(480, amp=20))
        self.assertEqual(pipeline.session.current_partial_text, "")
        self.assertGreaterEqual(len(pipeline.session.sentences), 1)
        rec = next(iter(pipeline.session.sentences.values()))
        self.assertIn("嗯", rec.provisional_text)
        msgs = _drain(pipeline.output_queue)
        self.assertTrue(any(m.get("type") == "provisional" for m in msgs))
        self.assertGreater(pipeline.session.last_committed_end_ms, 0)

    async def test_watchdog_leaked_speech_recovered(self):
        # VAD 漏检的真实语音同样攒字触发看门狗 → 强切进 Qwen 救回
        streaming = _DelayedBurstStreaming(start_after_ms=3500, token="漏检语音一二三四五")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        await self._feed(pipeline, _pcm_tone(4000, amp=8000))
        self.assertEqual(pipeline.session.current_partial_text, "")
        self.assertGreaterEqual(len(pipeline.session.sentences), 1)
        rec = next(iter(pipeline.session.sentences.values()))
        self.assertIn("漏检", rec.provisional_text)

    async def test_commit_leftover_goes_to_guard(self):
        # COMMIT 残留不再本地丢弃, 一律走 endpoint 让 Qwen+Guard 裁决
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        pipeline.session.current_partial_text = "嗯嗯嗯嗯嗯嗯嗯嗯"
        pipeline.session.is_in_speech = False
        pipeline.session.ring_buffer.write(_pcm_tone(500, amp=20))
        await pipeline.process_commit()
        self.assertEqual(pipeline.session.current_partial_text, "")
        self.assertGreaterEqual(len(pipeline.session.sentences), 1)

    async def test_watchdog_rate_limit_discards(self):
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        now = time.time()
        pipeline.session.watchdog_force_times.extend([now] * 20)
        await self._feed(pipeline, _pcm_tone(480, amp=20))
        self.assertEqual(pipeline.session.current_partial_text, "")
        self.assertEqual(pipeline.session.sentences, {})
        msgs = _drain(pipeline.output_queue)
        self.assertEqual([m for m in msgs if m.get("type") in ("provisional", "final")], [])

    async def test_stop_recovers_leftover(self):
        # STOP 是漏检语音的最后救回机会: 残留进 endpoint, 不本地丢
        streaming = _TokenStreaming("嗯")
        pipeline = self._pipeline(_SilentVAD(), streaming)
        pipeline.session.current_partial_text = "嗯嗯嗯嗯嗯嗯嗯嗯"
        pipeline.session.ring_buffer.write(_pcm_tone(500, amp=20))
        await pipeline.process_stop()
        self.assertGreaterEqual(len(pipeline.session.sentences), 1)
        rec = next(iter(pipeline.session.sentences.values()))
        self.assertIn("嗯", rec.provisional_text)


class TestNgramCanary(unittest.TestCase):
    def test_looping_text_is_flagged(self):
        from core.metrics.prometheus_metrics import metrics

        cfg = _base_config()
        session = ClientSession(session_id="canary", app_config=cfg)
        pipe = object.__new__(SessionPipeline)
        pipe.session = session
        pipe.config = cfg
        before = metrics._counters.get("asr_degenerate_ngram_detected_total", 0.0)
        pipe._note_outgoing_text("谢谢大家谢谢大家谢谢大家", "partial")
        after = metrics._counters.get("asr_degenerate_ngram_detected_total", 0.0)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
