"""COMMIT 清 VAD cache 后时间轴不得回跳。

根因: process_commit 把 vad_cache 换成空 dict 后, FunASR/Mock VAD 都从 0
重计毫秒; fsmn_vad 把返回值原样当 ring-buffer 绝对时间, 后续句 start 回跳
到 COMMIT 之前。叠加 force-cut (current_ms - 回跳后的 speech_start >=
max_speech) 与 process_stop 收尾缺水位钳制, 产生重复转写和空文本 final.

单会话即可复现, 与并发无关。本文件用 Mock VAD 锁死该契约。
"""
import asyncio
import math
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from config.settings import AppConfig, VADConfig
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue
from core.guard.consistency_guard import ConsistencyGuard
from core.hotword.hotword_manager import HotwordManager
from core.itn.itn_processor import ITNProcessor
from core.session import ClientSession
from core.speaker.mock_speaker import MockSpeakerEngine
from core.streaming_asr.mock_streaming import MockStreamingASREngine
from core.vad.mock_vad import MockVADEngine
from pipeline.session_pipeline import SessionPipeline


def _pcm_tone(duration_ms: int, freq: float = 260.0, amp: int = 8000) -> bytes:
    n = duration_ms * 16  # 16 kHz mono
    samples = [
        int(amp * math.sin(2 * math.pi * freq * i / 16000.0)) for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


def _pcm_silence(duration_ms: int) -> bytes:
    return b"\x00\x00" * (duration_ms * 16)


def _drain(queue: asyncio.Queue):
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    return messages


class TestMockVADCacheContract(unittest.TestCase):
    """Engine-level: empty cache restarts the VAD clock at 0."""

    def test_cache_reset_restarts_clock(self):
        vad = MockVADEngine(VADConfig(), energy_threshold=50.0)
        cache = {}
        speech = _pcm_tone(400)
        vad.process_chunk(speech, cache)
        t1 = cache["current_time_ms"]
        self.assertEqual(t1, 400)

        cache.clear()
        vad.process_chunk(speech, cache)
        self.assertEqual(cache["current_time_ms"], 400)

        # Without reset the clock would be 800ms; empty cache must not continue it
        cache_kept = {}
        vad.process_chunk(speech, cache_kept)
        vad.process_chunk(speech, cache_kept)
        self.assertEqual(cache_kept["current_time_ms"], 800)


class TestMidSpeechCommitTimeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AppConfig()
        self.config.streaming_asr.backend = "mock"
        self.config.vad.backend = "mock"
        self.config.final_asr.engine_type = "mock"
        self.config.speaker.backend = "mock"
        self.config.vad.max_end_silence_time = 200
        # Keep force-cut out of the VAD-endpoint case
        self.config.vad.max_speech_duration_ms = 60000
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.final_queue = FinalQueue(
            MockQwenEngine(self.config.final_asr, simulated_latency_ms=1.0),
            self.config.final_asr,
        )
        self.final_queue.start()

    async def asyncTearDown(self):
        await self.final_queue.stop()
        self.thread_pool.shutdown(wait=False)

    def _make_pipeline(self, session_id: str, max_speech_ms: int = 0) -> SessionPipeline:
        if max_speech_ms:
            self.config.vad.max_speech_duration_ms = max_speech_ms
        session = ClientSession(session_id=session_id, app_config=self.config)
        output_queue = asyncio.Queue()
        pipeline = SessionPipeline(
            session=session,
            vad_engine=MockVADEngine(self.config.vad, energy_threshold=50.0),
            streaming_asr_engine=MockStreamingASREngine(self.config.streaming_asr),
            final_queue=self.final_queue,
            speaker_engine=MockSpeakerEngine(self.config.speaker),
            punc_engine=None,
            guard=ConsistencyGuard(self.config.guard),
            itn=ITNProcessor(self.config.itn),
            hotword_mgr=HotwordManager(),
            thread_pool=self.thread_pool,
            output_queue=output_queue,
            config=self.config,
        )
        return pipeline

    async def _feed(self, pipeline: SessionPipeline, pcm: bytes, chunk_ms: int = 60):
        chunk_bytes = chunk_ms * 32
        for offset in range(0, len(pcm), chunk_bytes):
            await pipeline.process_audio_chunk(pcm[offset : offset + chunk_bytes])

    def _assert_no_overlap(self, sentences):
        ordered = sorted(sentences, key=lambda r: r.sentence_id)
        for prev, cur in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(
                cur.start_ms,
                prev.end_ms,
                f"overlap [{prev.start_ms}~{prev.end_ms}] vs [{cur.start_ms}~{cur.end_ms}]",
            )

    async def test_mid_speech_commit_then_vad_endpoint_does_not_rewind(self):
        """说话中途 COMMIT 后, 后续 VAD 句起点必须落在 COMMIT 时刻之后."""
        pipeline = self._make_pipeline("commit-vad")
        session = pipeline.session

        await self._feed(pipeline, _pcm_tone(2000))
        commit_ms = session.ring_buffer.total_ms
        self.assertTrue(session.is_in_speech)
        await pipeline.process_commit()

        self.assertEqual(session.vad_timeline_origin_ms, commit_ms)
        self.assertEqual(session.vad_cache, {})
        self.assertGreaterEqual(session.last_committed_end_ms, commit_ms)

        # New utterance after COMMIT, then silence so Mock VAD emits an endpoint
        await self._feed(pipeline, _pcm_tone(800))
        await self._feed(pipeline, _pcm_silence(400))
        await pipeline.process_stop()

        sentences = sorted(session.sentences.values(), key=lambda r: r.sentence_id)
        self.assertGreaterEqual(len(sentences), 2)
        self._assert_no_overlap(sentences)

        commit_sentence = sentences[0]
        self.assertEqual(commit_sentence.end_ms, commit_ms)
        for rec in sentences[1:]:
            self.assertGreaterEqual(
                rec.start_ms,
                commit_ms,
                f"post-COMMIT sentence rewound: [{rec.start_ms}~{rec.end_ms}] "
                f"commit_ms={commit_ms}",
            )
            self.assertGreater(rec.end_ms, rec.start_ms)

        finals = [m for m in _drain(pipeline.output_queue) if m.get("type") == "final"]
        for msg in finals:
            self.assertTrue(
                (msg["end_ms"] - msg["start_ms"]) > 0 or msg.get("text") == "",
            )
            if msg["end_ms"] > commit_ms:
                self.assertGreaterEqual(msg["start_ms"], commit_ms)

    async def test_mid_speech_commit_does_not_immediate_force_cut(self):
        """COMMIT 后 VAD start 若仍是 0, force-cut 会立刻把整段 [0, now] 再切一遍."""
        pipeline = self._make_pipeline("commit-force-cut", max_speech_ms=1500)
        session = pipeline.session

        await self._feed(pipeline, _pcm_tone(2000))
        commit_ms = session.ring_buffer.total_ms
        await pipeline.process_commit()
        n_after_commit = len(session.sentences)

        # 800ms more speech: well under 1500ms max_speech if the clock is rebased
        await self._feed(pipeline, _pcm_tone(800))
        n_before_stop = len(session.sentences)
        self.assertEqual(
            n_before_stop,
            n_after_commit,
            "force-cut fired immediately after COMMIT (VAD clock still at 0)",
        )

        await pipeline.process_stop()
        sentences = sorted(session.sentences.values(), key=lambda r: r.sentence_id)
        self._assert_no_overlap(sentences)
        # Anything that extends past COMMIT must start at/after the COMMIT instant
        for rec in sentences:
            if rec.end_ms > commit_ms:
                self.assertGreaterEqual(rec.start_ms, commit_ms)
            self.assertGreater(rec.end_ms, rec.start_ms)

    async def test_handle_sentence_endpoint_clamps_and_skips_overlap(self):
        pipeline = self._make_pipeline("clamp")
        session = pipeline.session
        session.last_committed_end_ms = 5000

        await pipeline._handle_sentence_endpoint(0, 3000)
        self.assertEqual(session.sentences, {})
        self.assertEqual(session.last_committed_end_ms, 5000)

        await pipeline._handle_sentence_endpoint(1000, 8000)
        self.assertEqual(len(session.sentences), 1)
        rec = next(iter(session.sentences.values()))
        self.assertEqual(rec.start_ms, 5000)
        self.assertEqual(rec.end_ms, 8000)
        self.assertEqual(session.last_committed_end_ms, 8000)


if __name__ == "__main__":
    unittest.main()
