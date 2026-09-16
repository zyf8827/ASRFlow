import asyncio
import unittest
from config.settings import FinalASRConfig
from core.final_asr.mock_qwen import MockQwenEngine
from core.final_asr.qwen_queue import FinalQueue


class TestFinalQueue(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = FinalASRConfig(
            engine_type="mock",
            max_batch_size=4,
            batch_window_ms=10,
            max_concurrency=4,
            hard_timeout_sec=1.5,
        )
        self.engine = MockQwenEngine(self.config, simulated_latency_ms=20.0)
        self.queue = FinalQueue(self.engine, self.config)
        self.queue.start()

    async def asyncTearDown(self):
        await self.queue.stop()

    async def test_single_submission(self):
        audio = b"\x00\x01" * 16000  # 1s audio
        res = await self.queue.submit(audio, context="你好昨天晚上睡得怎么样")
        self.assertEqual(res, "你好，昨天晚上睡得怎么样？")
        self.assertEqual(self.queue.total_jobs_completed, 1)

    async def test_concurrent_batching(self):
        audio = b"\x00\x01" * 16000
        contexts = [
            "今天天气不错我们下午去公园走走吧",
            "我把会议改到明天上午十点了",
            "地铁有点挤我可能要晚到十分钟",
            "晚上一起吃饭你想吃火锅还是日料",
        ]

        tasks = [self.queue.submit(audio, context=c) for c in contexts]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0], "今天天气不错，我们下午去公园走走吧。")
        self.assertEqual(results[1], "我把会议改到明天上午十点了。")
        self.assertEqual(self.queue.total_jobs_completed, 4)


class TestPerItemConcurrency(unittest.IsolatedAsyncioTestCase):
    """max_concurrency must cap in-flight requests per ITEM, not per batch."""

    class _TrackingEngine(MockQwenEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inflight = 0
            self.max_inflight = 0

        async def transcribe(self, audio_bytes, context=None, hotwords=None):
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            try:
                return await super().transcribe(audio_bytes, context=context, hotwords=hotwords)
            finally:
                self.inflight -= 1

    async def test_concurrency_capped_per_item(self):
        config = FinalASRConfig(
            engine_type="mock",
            max_batch_size=8,
            batch_window_ms=10,
            max_concurrency=2,
            hard_timeout_sec=5.0,
        )
        engine = self._TrackingEngine(config, simulated_latency_ms=50.0)
        queue = FinalQueue(engine, config)
        queue.start()
        try:
            audio = b"\x00\x01" * 16000
            results = await asyncio.gather(
                *[queue.submit(audio, context=f"句子{i}") for i in range(8)]
            )
            self.assertEqual(len(results), 8)
            self.assertTrue(all(isinstance(r, str) for r in results))
            # Cap enforced: at most max_concurrency requests in flight at once,
            # and the cap is actually reachable (not serialized down to 1)
            self.assertLessEqual(engine.max_inflight, 2)
            self.assertGreaterEqual(engine.max_inflight, 2)
        finally:
            await queue.stop()


if __name__ == "__main__":
    unittest.main()
