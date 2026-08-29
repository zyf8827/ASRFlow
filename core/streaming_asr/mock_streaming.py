import math
import struct
from typing import Dict, Any, List, Optional

from core.streaming_asr.base import BaseStreamingASREngine
from config.settings import StreamingASRConfig


class MockStreamingASREngine(BaseStreamingASREngine):
    """
    Mock streaming ASR engine that generates progressive text tokens
    simulating real-time partial output.
    """

    MOCK_SENTENCES = [
        "你好昨天晚上睡得怎么样",
        "今天天气不错我们下午去公园走走吧",
        "我把会议改到明天上午十点了",
        "晚上一起吃饭你想吃火锅还是日料",
        "地铁有点挤我可能要晚到十分钟",
        "周末有空的话我们去看新上映的电影",
    ]

    def __init__(self, config: StreamingASRConfig):
        self.config = config

    def _calc_rms(self, audio_bytes: bytes) -> float:
        if not audio_bytes or len(audio_bytes) < 2:
            return 0.0
        sample_count = len(audio_bytes) // 2
        try:
            samples = struct.unpack(f"<{sample_count}h", audio_bytes[: sample_count * 2])
            sum_sq = sum(s * s for s in samples)
            return math.sqrt(sum_sq / sample_count)
        except Exception:
            return 0.0

    def process_chunk(
        self,
        audio_bytes: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        hotwords: Optional[List[str]] = None,
    ) -> str:
        accumulated_ms = cache.get("accumulated_ms", 0)
        chunk_ms = len(audio_bytes) // 32 if audio_bytes else 0
        accumulated_ms += chunk_ms
        cache["accumulated_ms"] = accumulated_ms

        sentence_idx = cache.get("mock_idx", 0)
        target_text = self.MOCK_SENTENCES[sentence_idx % len(self.MOCK_SENTENCES)]

        if hotwords and len(hotwords) > 0 and (sentence_idx % 2 == 0):
            target_text = f"我们晚点去{hotwords[0]}碰头吧"

        emitted = cache.get("emitted_chars", 0)

        if is_final:
            cache["mock_idx"] = sentence_idx + 1
            cache["accumulated_ms"] = 0
            cache["emitted_chars"] = 0
            return target_text[emitted:]

        # Return only newly revealed chars (~3 chars per second), mirroring the
        # incremental per-chunk delta semantics of the real Paraformer engine
        char_count = min(len(target_text), max(1, accumulated_ms // 300))
        cache["emitted_chars"] = char_count
        return target_text[emitted:char_count]

    def flush(
        self, cache: Dict[str, Any], hotwords: Optional[List[str]] = None
    ) -> str:
        return self.process_chunk(b"", cache=cache, is_final=True, hotwords=hotwords)
