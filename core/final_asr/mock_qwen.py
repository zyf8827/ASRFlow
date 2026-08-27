import asyncio
import time
from typing import Dict, Any, List, Optional

from core.final_asr.base import BaseFinalASREngine
from config.settings import FinalASRConfig


class MockQwenEngine(BaseFinalASREngine):
    """
    Mock Qwen3-ASR Engine for development and tests.

    `context` is prompt history (S3) and must not be copied as this sentence's
    transcript. Exact keys in MOCK_CORRECTIONS still map to refined text so
    queue tests can assert per-job result routing.
    """

    MOCK_CORRECTIONS = {
        "你好昨天晚上睡得怎么样": "你好，昨天晚上睡得怎么样？",
        "今天天气不错我们下午去公园走走吧": "今天天气不错，我们下午去公园走走吧。",
        "我把会议改到明天上午十点了": "我把会议改到明天上午十点了。",
        "晚上一起吃饭你想吃火锅还是日料": "晚上一起吃饭，你想吃火锅还是日料？",
        "地铁有点挤我可能要晚到十分钟": "地铁有点挤，我可能要晚到十分钟。",
        "周末有空的话我们去看新上映的电影": "周末有空的话，我们去看新上映的电影。",
    }

    def __init__(self, config: FinalASRConfig, simulated_latency_ms: float = 80.0):
        self.config = config
        self.simulated_latency_ms = simulated_latency_ms

    async def transcribe(
        self,
        audio_bytes: bytes,
        context: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        # Simulate Ascend NPU computation delay
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        if not audio_bytes:
            return ""

        if context:
            refined = self.MOCK_CORRECTIONS.get(context.strip())
            if refined is not None:
                return refined

        return "嗯好的，那我们到时候见。"

    async def transcribe_batch(
        self, batch_items: List[Dict[str, Any]]
    ) -> List[Any]:
        # Use gather semantics so tests can cover per-job failure (mirrors real engine)
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        coros = [
            self.transcribe(
                item["audio_bytes"],
                context=item.get("context"),
                hotwords=item.get("hotwords"),
            )
            for item in batch_items
        ]
        return await asyncio.gather(*coros, return_exceptions=True)
