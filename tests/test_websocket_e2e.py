import json
import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
import websockets

from config.settings import AppConfig
from core.session import ClientSession
from core.session_manager import SessionManager
from core.vad.mock_vad import MockVADEngine
from core.streaming_asr.mock_streaming import MockStreamingASREngine
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue
from core.speaker.mock_speaker import MockSpeakerEngine
from core.guard.consistency_guard import ConsistencyGuard
from core.itn.itn_processor import ITNProcessor
from core.hotword.hotword_manager import HotwordManager
from pipeline.session_pipeline import SessionPipeline
from server.websocket_server import WebSocketGateway
from demo.client_demo import generate_synthetic_audio as generate_synthetic_speech_pcm


class _MockServiceHarness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AppConfig()
        self.config.server.port = 10199
        self.config.server.http_port = 10198
        self.config.streaming_asr.backend = "mock"
        self.config.vad.backend = "mock"
        self.config.final_asr.engine_type = "mock"
        self.config.speaker.backend = "mock"

        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.vad_engine = MockVADEngine(self.config.vad, energy_threshold=50.0)
        self.streaming_engine = MockStreamingASREngine(self.config.streaming_asr)
        self.final_engine = MockQwenEngine(self.config.final_asr, simulated_latency_ms=10.0)
        self.final_queue = FinalQueue(self.final_engine, self.config.final_asr)
        self.final_queue.start()

        self.speaker_engine = MockSpeakerEngine(self.config.speaker)
        self.guard = ConsistencyGuard(self.config.guard)
        self.itn = ITNProcessor(self.config.itn)
        self.hotword_mgr = HotwordManager()
        self.session_manager = SessionManager(self.config)

        def pipeline_factory(session: ClientSession, output_queue: asyncio.Queue) -> SessionPipeline:
            return SessionPipeline(
                session=session,
                vad_engine=self.vad_engine,
                streaming_asr_engine=self.streaming_engine,
                final_queue=self.final_queue,
                speaker_engine=self.speaker_engine,
                punc_engine=None,
                guard=self.guard,
                itn=self.itn,
                hotword_mgr=self.hotword_mgr,
                thread_pool=self.thread_pool,
                output_queue=output_queue,
                config=self.config,
            )

        self.gateway = WebSocketGateway(
            session_manager=self.session_manager,
            pipeline_factory=pipeline_factory,
            config=self.config,
            admission=getattr(self, "admission", None),
        )
        await self.gateway.start()

    async def asyncTearDown(self):
        await self.gateway.stop()
        await self.final_queue.stop()
        self.thread_pool.shutdown(wait=False)


class TestWebSocketE2E(_MockServiceHarness):
    async def test_full_session_flow(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            # 1. Ping / Pong
            await ws.send("ping")
            pong = await ws.recv()
            self.assertEqual(json.loads(pong).get("type"), "pong")

            # 2. START command
            start_cmd = {
                "type": "start",
                "session_id": "test-e2e-session-1",
                "language": "zh",
                "enable_spk": True,
                "expected_speakers": 2,
                "hotwords": ["星巴克", "电影院"],
            }
            await ws.send(json.dumps(start_cmd))
            ready_msg = await ws.recv()
            ready_data = json.loads(ready_msg)
            self.assertEqual(ready_data.get("type"), "session_ready")
            self.assertEqual(ready_data.get("session_id"), "test-e2e-session-1")

            # 3. Stream Synthetic Speech (3 seconds = 50 chunks of 60ms)
            audio = generate_synthetic_speech_pcm(duration_sec=3.0)
            chunk_size = 1920
            received_partials = []
            received_finals = []
            received_provisionals = []

            async def collector():
                async for msg in ws:
                    d = json.loads(msg)
                    mtype = d.get("type")
                    if mtype == "partial":
                        received_partials.append(d)
                    elif mtype == "provisional":
                        received_provisionals.append(d)
                    elif mtype == "final":
                        received_finals.append(d)
                    elif mtype == "session_finished":
                        break

            collect_task = asyncio.create_task(collector())

            # Stream audio frames
            offset = 0
            while offset < len(audio):
                chunk = audio[offset : offset + chunk_size]
                offset += len(chunk)
                await ws.send(chunk)
                await asyncio.sleep(0.01)

            # Send silence frames to trigger VAD endpoint
            silence = b"\x00" * (16000 * 2)  # 1s silence
            offset = 0
            while offset < len(silence):
                chunk = silence[offset : offset + chunk_size]
                offset += len(chunk)
                await ws.send(chunk)
                await asyncio.sleep(0.01)

            # Send STOP
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(collect_task, timeout=5.0)

            # Verify received results
            self.assertGreater(len(received_partials), 0)
            self.assertGreater(len(received_provisionals), 0)
            self.assertGreater(len(received_finals), 0)

            first_final = received_finals[0]
            self.assertIn("text", first_final)
            self.assertIn("speaker", first_final)
            self.assertIn("final_source", first_final)
            self.assertEqual(first_final.get("session_id"), "test-e2e-session-1")

    async def test_funasr_text_protocol_flow(self):
        """Test plain text commands: START, HOTWORDS:..., COMMIT, STOP."""
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            # 1. Plain START
            await ws.send("START")
            ready_msg = await ws.recv()
            ready_data = json.loads(ready_msg)
            self.assertEqual(ready_data.get("event"), "started")

            # 2. HOTWORDS
            await ws.send("HOTWORDS:星巴克,电影院")
            hw_msg = await ws.recv()
            hw_data = json.loads(hw_msg)
            self.assertEqual(hw_data.get("event"), "hotwords_set")
            self.assertEqual(hw_data.get("hotwords"), ["星巴克", "电影院"])

            # 3. Stream 1.5s audio
            audio = generate_synthetic_speech_pcm(duration_sec=1.5)
            await ws.send(audio)
            await asyncio.sleep(0.05)

            # 4. Manual COMMIT
            await ws.send("COMMIT")
            await asyncio.sleep(0.1)

            # 5. STOP
            await ws.send("STOP")
            stop_msg = await ws.recv()
            # Can receive partial, provisional, final, or stopped
            self.assertTrue(len(stop_msg) > 0)


class TestMaxSpeechDurationForceCut(unittest.IsolatedAsyncioTestCase):
    """Continuous speech without VAD pauses must be force-cut at max_speech_duration_ms."""

    async def test_force_cut_splits_continuous_speech(self):
        config = AppConfig()
        config.streaming_asr.backend = "mock"
        config.vad.backend = "mock"
        config.final_asr.engine_type = "mock"
        config.speaker.backend = "mock"
        config.vad.max_speech_duration_ms = 1500

        thread_pool = ThreadPoolExecutor(max_workers=2)
        final_queue = FinalQueue(
            MockQwenEngine(config.final_asr, simulated_latency_ms=5.0), config.final_asr
        )
        final_queue.start()

        try:
            session = ClientSession(session_id="force-cut-test", app_config=config)
            output_queue = asyncio.Queue()
            pipeline = SessionPipeline(
                session=session,
                vad_engine=MockVADEngine(config.vad, energy_threshold=50.0),
                streaming_asr_engine=MockStreamingASREngine(config.streaming_asr),
                final_queue=final_queue,
                speaker_engine=MockSpeakerEngine(config.speaker),
                punc_engine=None,
                guard=ConsistencyGuard(config.guard),
                itn=ITNProcessor(config.itn),
                hotword_mgr=HotwordManager(),
                thread_pool=thread_pool,
                output_queue=output_queue,
                config=config,
            )

            # 4s of continuous speech (no silence -> Mock VAD never ends it)
            audio = generate_synthetic_speech_pcm(duration_sec=4.0)
            chunk_size = 1920  # 60ms
            for offset in range(0, len(audio), chunk_size):
                await pipeline.process_audio_chunk(audio[offset : offset + chunk_size])
            await pipeline.process_stop()

            messages = []
            while not output_queue.empty():
                messages.append(output_queue.get_nowait())
            provisionals = [m for m in messages if m.get("type") == "provisional"]
            finals = [m for m in messages if m.get("type") == "final"]

            # Force cuts at ~1.5s / ~3.0s. Synthetic audio goes silent at 3s, so
            # the VAD endpoint coincides with the second cut and is skipped as
            # a zero-length overlap rather than emitted as a third empty sentence.
            self.assertGreaterEqual(len(provisionals), 2)
            self.assertGreaterEqual(len(finals), 2)
            for m in finals:
                self.assertGreater(m["end_ms"] - m["start_ms"], 0)
                self.assertLessEqual(m["end_ms"] - m["start_ms"], 1600)
        finally:
            await final_queue.stop()
            thread_pool.shutdown(wait=False)


class TestSessionResume(_MockServiceHarness):
    """A disconnected session must be resumable within the grace period."""

    async def test_resume_after_abrupt_disconnect(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        session_id = "resume-test-session-1"

        # Connection 1: stream audio, get a finalized sentence, then drop
        finals_before_disconnect = []
        async with websockets.connect(uri) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "session_id": session_id,
                        "language": "zh",
                        "enable_spk": True,
                        "expected_speakers": 2,
                    }
                )
            )
            ready = json.loads(await ws.recv())
            self.assertEqual(ready.get("type"), "session_ready")

            async def collector():
                async for msg in ws:
                    d = json.loads(msg)
                    if d.get("type") == "final":
                        finals_before_disconnect.append(d)

            collect_task = asyncio.create_task(collector())
            audio = generate_synthetic_speech_pcm(duration_sec=3.0)
            silence = b"\x00" * (16000 * 2)  # 1s
            for chunk_offset in range(0, len(audio), 1920):
                await ws.send(audio[chunk_offset : chunk_offset + 1920])
                await asyncio.sleep(0.01)
            for chunk_offset in range(0, len(silence), 1920):
                await ws.send(silence[chunk_offset : chunk_offset + 1920])
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.5)  # let the final arrive
            collect_task.cancel()
            # No STOP: simulate an abrupt disconnect

        self.assertGreaterEqual(len(finals_before_disconnect), 1)
        await asyncio.sleep(0.5)  # allow server-side detach to complete

        # Connection 2: resume the same session
        async with websockets.connect(uri) as ws2:
            await ws2.send(
                json.dumps({"type": "start", "session_id": session_id, "resume": True})
            )
            resumed = json.loads(await asyncio.wait_for(ws2.recv(), timeout=3.0))
            self.assertEqual(resumed.get("type"), "session_resumed")
            self.assertEqual(resumed.get("session_id"), session_id)
            self.assertGreaterEqual(resumed.get("total_sentences"), 1)
            self.assertGreater(resumed.get("last_end_ms"), 0)

            # The resumed session keeps working: stream more audio and stop
            async def collector2():
                async for msg in ws2:
                    d = json.loads(msg)
                    if d.get("type") == "session_finished":
                        return d
                    if d.get("type") == "final":
                        finals_before_disconnect.append(d)

            collect_task2 = asyncio.create_task(collector2())
            for chunk_offset in range(0, len(audio), 1920):
                await ws2.send(audio[chunk_offset : chunk_offset + 1920])
                await asyncio.sleep(0.01)
            await ws2.send(json.dumps({"type": "stop"}))
            finished = await asyncio.wait_for(collect_task2, timeout=5.0)
            # Previous committed sentences are preserved in the final transcript
            self.assertGreaterEqual(finished.get("total_sentences", 0), 1)


class TestSessionResumeExpired(_MockServiceHarness):
    """Resuming after the grace period expired must equal a brand-new session."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config.server.resume_ttl_sec = 0.2  # very short grace period

    async def test_expired_resume_creates_new_session(self):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        session_id = "expired-resume-session-1"

        # Connection 1: produce one committed sentence, then drop without STOP
        async with websockets.connect(uri) as ws:
            await ws.send(
                json.dumps({"type": "start", "session_id": session_id, "enable_spk": False})
            )
            ready = json.loads(await ws.recv())
            self.assertEqual(ready.get("type"), "session_ready")
            audio = generate_synthetic_speech_pcm(duration_sec=3.0)
            silence = b"\x00" * (16000 * 2)  # 1s
            for data in (audio, silence):
                for chunk_offset in range(0, len(data), 1920):
                    await ws.send(data[chunk_offset : chunk_offset + 1920])
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.4)
        # Outlive the resume TTL (cleaner interval is 60s, so the expired
        # session is still parked in the manager at this point)
        await asyncio.sleep(0.6)

        # Connection 2: resume request after expiry -> brand-new session
        async with websockets.connect(uri) as ws2:
            await ws2.send(
                json.dumps({"type": "start", "session_id": session_id, "resume": True})
            )
            msg = json.loads(await asyncio.wait_for(ws2.recv(), timeout=3.0))
            self.assertEqual(msg.get("type"), "session_ready")
            # The new session works normally and is unbound from the old state
            async def collector2():
                async for m in ws2:
                    d = json.loads(m)
                    if d.get("type") == "session_finished":
                        return d

            collect_task = asyncio.create_task(collector2())
            for chunk_offset in range(0, len(audio), 1920):
                await ws2.send(audio[chunk_offset : chunk_offset + 1920])
                await asyncio.sleep(0.01)
            await ws2.send(json.dumps({"type": "stop"}))
            finished = await asyncio.wait_for(collect_task, timeout=5.0)
            self.assertIsNotNone(finished)
            # Fresh session: audio timeline restarted at 0 (only this
            # connection's 3s), a resumed session would carry >= 7s
            self.assertLess(finished.get("duration_ms", 0), 4000)


class TestMalformedFrameTolerance(_MockServiceHarness):
    """Malformed binary frames must be dropped/truncated, never kill the connection."""

    async def _run_session_with_frames(self, frames):
        uri = f"ws://127.0.0.1:{self.config.server.port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "start", "session_id": "malformed-frame-test"}))
            ready = json.loads(await ws.recv())
            self.assertEqual(ready.get("type"), "session_ready")

            async def collector():
                async for msg in ws:
                    d = json.loads(msg)
                    if d.get("type") == "session_finished":
                        return d

            collect_task = asyncio.create_task(collector())
            for frame in frames:
                await ws.send(frame)
                await asyncio.sleep(0.01)
            audio = generate_synthetic_speech_pcm(duration_sec=1.0)
            silence = b"\x00" * 32000  # 1s
            for data in (audio, silence):
                for offset in range(0, len(data), 1920):
                    await ws.send(data[offset : offset + 1920])
                    await asyncio.sleep(0.01)
            await ws.send(json.dumps({"type": "stop"}))
            return await asyncio.wait_for(collect_task, timeout=5.0)

    async def test_odd_length_frame_does_not_kill_connection(self):
        # Half a sample: must be truncated by the gateway, not crash the engines
        finished = await self._run_session_with_frames([b"\x11" * 1921])
        self.assertGreaterEqual(finished.get("total_sentences", 0), 0)

    async def test_oversized_frame_truncated_connection_alive(self):
        # 70000B = 2.1875s audio > default 2000ms cap -> truncated, session lives on
        finished = await self._run_session_with_frames([b"\x00" * 70000])
        self.assertGreaterEqual(finished.get("total_sentences", 0), 0)

    async def test_garbage_byte_frame_survives(self):
        finished = await self._run_session_with_frames([b"\xff" * 640])
        self.assertGreaterEqual(finished.get("total_sentences", 0), 0)


class TestFrameValidation(unittest.TestCase):
    """Unit tests for the gateway binary-frame gate."""

    def _gateway(self, max_frame_ms=None):
        config = AppConfig()
        if max_frame_ms is not None:
            config.server.max_audio_frame_ms = max_frame_ms
        return WebSocketGateway(
            session_manager=None, pipeline_factory=None, config=config
        )

    def test_odd_length_truncated_to_even(self):
        frame = self._gateway()._validate_audio_frame(b"\x11" * 1921)
        self.assertEqual(len(frame), 1920)

    def test_single_byte_frame_becomes_none(self):
        self.assertIsNone(self._gateway()._validate_audio_frame(b"\x11"))

    def test_empty_frame_is_none(self):
        self.assertIsNone(self._gateway()._validate_audio_frame(b""))

    def test_oversized_truncated_to_limit(self):
        gateway = self._gateway(max_frame_ms=100)  # 100ms = 3200 bytes
        frame = gateway._validate_audio_frame(b"\x00" * 9999)
        self.assertEqual(len(frame), 3200)

    def test_normal_frame_passthrough(self):
        data = b"\x11" * 1920
        self.assertEqual(self._gateway()._validate_audio_frame(data), data)


class TestFinalQueueOverflow(unittest.IsolatedAsyncioTestCase):
    async def test_queue_full_rejects_submit(self):
        config = AppConfig()
        config.final_asr.max_queue_size = 1
        config.final_asr.hard_timeout_sec = 0.2

        queue = FinalQueue(
            MockQwenEngine(config.final_asr, simulated_latency_ms=10.0), config.final_asr
        )
        # Pretend the dispatcher is running but never drains
        queue._is_running = True

        # First job occupies the single queue slot (its future never resolves)
        with self.assertRaises(asyncio.TimeoutError):
            await queue.submit(b"\x00" * 3200, timeout_sec=0.2)

        # Second job must be rejected, not queued
        with self.assertRaises(RuntimeError):
            await queue.submit(b"\x00" * 3200, timeout_sec=0.2)
        self.assertEqual(queue._queue.qsize(), 1)


class TestHotwordRefresh(unittest.IsolatedAsyncioTestCase):
    async def test_update_hotwords_applies_to_both_passes(self):
        config = AppConfig()
        config.streaming_asr.backend = "mock"
        config.vad.backend = "mock"
        config.final_asr.engine_type = "mock"
        config.speaker.backend = "mock"

        thread_pool = ThreadPoolExecutor(max_workers=2)
        final_queue = FinalQueue(
            MockQwenEngine(config.final_asr, simulated_latency_ms=5.0), config.final_asr
        )
        final_queue.start()
        try:
            session = ClientSession(session_id="hotword-test", app_config=config)
            pipeline = SessionPipeline(
                session=session,
                vad_engine=MockVADEngine(config.vad, energy_threshold=50.0),
                streaming_asr_engine=MockStreamingASREngine(config.streaming_asr),
                final_queue=final_queue,
                speaker_engine=MockSpeakerEngine(config.speaker),
                punc_engine=None,
                guard=ConsistencyGuard(config.guard),
                itn=ITNProcessor(config.itn),
                hotword_mgr=HotwordManager(),
                thread_pool=thread_pool,
                output_queue=asyncio.Queue(),
                config=config,
            )

            self.assertEqual(pipeline._hotwords, [])
            pipeline.update_hotwords(["星巴克", "电影院", "星巴克"])
            # session keeps the raw list (Qwen prompt); Pass-1 biasing is deduped
            self.assertEqual(session.hotwords, ["星巴克", "电影院", "星巴克"])
            self.assertEqual(pipeline._hotwords, ["星巴克", "电影院"])
        finally:
            await final_queue.stop()
            thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
