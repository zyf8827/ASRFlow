import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

from config.settings import AppConfig, FinalASRConfig
from core.final_asr.qwen_engine import Qwen3ASREngine
from core.guard.consistency_guard import ConsistencyGuard
from core.hotword.hotword_manager import HotwordManager
from core.itn.itn_processor import ITNProcessor
from core.session import ClientSession, SentenceRecord
from core.speaker.mock_speaker import MockSpeakerEngine
from core.streaming_asr.mock_streaming import MockStreamingASREngine
from core.vad.mock_vad import MockVADEngine
from pipeline.session_pipeline import SessionPipeline


class _CapturingFinalQueue:
    def __init__(self, reply: str = "二遍识别结果。"):
        self.reply = reply
        self.submitted = []

    async def submit(self, audio_bytes, context=None, hotwords=None, timeout_sec=None):
        self.submitted.append(
            {
                "audio_bytes": audio_bytes,
                "context": context,
                "hotwords": hotwords,
                "timeout_sec": timeout_sec,
            }
        )
        return self.reply


def _committed(sentence_id: int, final_text: str, provisional_text: str = "") -> SentenceRecord:
    return SentenceRecord(
        sentence_id=sentence_id,
        start_ms=sentence_id * 1000,
        end_ms=sentence_id * 1000 + 800,
        provisional_text=provisional_text or f"首遍{sentence_id}",
        final_text=final_text,
        is_committed=True,
    )


class TestFinalHistoryContext(unittest.TestCase):
    def _pipeline(self, history_chars: int = 200) -> SessionPipeline:
        config = AppConfig()
        config.final_asr.engine_type = "mock"
        config.final_asr.context_history_chars = history_chars
        config.speaker.backend = "mock"
        session = ClientSession(session_id="s3-ctx", enable_spk=False, app_config=config)
        return SessionPipeline(
            session=session,
            vad_engine=MockVADEngine(config.vad),
            streaming_asr_engine=MockStreamingASREngine(config.streaming_asr),
            final_queue=_CapturingFinalQueue(),
            speaker_engine=MockSpeakerEngine(config.speaker),
            punc_engine=None,
            guard=ConsistencyGuard(config.guard),
            itn=ITNProcessor(config.itn),
            hotword_mgr=HotwordManager(),
            thread_pool=ThreadPoolExecutor(max_workers=1),
            output_queue=asyncio.Queue(),
            config=config,
        )

    def test_empty_when_nothing_finalized(self):
        pipeline = self._pipeline()
        pipeline.session.sentences[1] = SentenceRecord(
            sentence_id=1, start_ms=0, end_ms=800, provisional_text="本句首遍"
        )
        self.assertEqual(pipeline._final_history_context(1), "")

    def test_uses_committed_finals_not_provisional(self):
        pipeline = self._pipeline()
        pipeline.session.sentences[1] = _committed(1, "正确的历史句甲。", "首遍错字甲")
        pipeline.session.sentences[2] = SentenceRecord(
            sentence_id=2, start_ms=1000, end_ms=1800, provisional_text="本句首遍错字"
        )
        ctx = pipeline._final_history_context(2)
        self.assertEqual(ctx, "正确的历史句甲。")
        self.assertNotIn("首遍", ctx)
        self.assertNotIn("本句", ctx)

    def test_skips_in_flight_previous_sentence(self):
        pipeline = self._pipeline()
        pipeline.session.sentences[1] = SentenceRecord(
            sentence_id=1,
            start_ms=0,
            end_ms=800,
            provisional_text="在途首遍",
            final_text="",
            is_committed=False,
        )
        pipeline.session.sentences[2] = SentenceRecord(
            sentence_id=2, start_ms=1000, end_ms=1800, provisional_text="本句首遍"
        )
        self.assertEqual(pipeline._final_history_context(2), "")

    def test_concatenates_and_keeps_tail(self):
        pipeline = self._pipeline(history_chars=10)
        pipeline.session.sentences[1] = _committed(1, "AAAAAAAAAA")
        pipeline.session.sentences[2] = _committed(2, "BBBBBBBBBB")
        pipeline.session.sentences[3] = SentenceRecord(
            sentence_id=3, start_ms=3000, end_ms=3800, provisional_text="本句"
        )
        self.assertEqual(pipeline._final_history_context(3), "BBBBBBBBBB")

    def test_zero_limit_disables_context(self):
        pipeline = self._pipeline(history_chars=0)
        pipeline.session.sentences[1] = _committed(1, "历史句。")
        self.assertEqual(pipeline._final_history_context(2), "")


class TestFinalHistoryContextSubmit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AppConfig()
        self.config.final_asr.engine_type = "mock"
        self.config.final_asr.context_history_chars = 200
        self.config.speaker.backend = "mock"
        self.queue = _CapturingFinalQueue()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.session = ClientSession(
            session_id="s3-submit", enable_spk=False, app_config=self.config
        )
        self.session.hotwords = ["星巴克"]
        self.pipeline = SessionPipeline(
            session=self.session,
            vad_engine=MockVADEngine(self.config.vad),
            streaming_asr_engine=MockStreamingASREngine(self.config.streaming_asr),
            final_queue=self.queue,
            speaker_engine=MockSpeakerEngine(self.config.speaker),
            punc_engine=None,
            guard=ConsistencyGuard(self.config.guard),
            itn=ITNProcessor(self.config.itn),
            hotword_mgr=HotwordManager(),
            thread_pool=self.thread_pool,
            output_queue=asyncio.Queue(),
            config=self.config,
        )

    async def asyncTearDown(self):
        self.thread_pool.shutdown(wait=False)

    async def test_submit_passes_history_not_provisional(self):
        self.session.sentences[1] = _committed(1, "正确的历史句甲。", "首遍错字甲")
        self.session.sentences[2] = SentenceRecord(
            sentence_id=2,
            start_ms=1000,
            end_ms=1800,
            provisional_text="本句首遍错字乙",
        )
        await self.pipeline._process_final_segment(
            sentence_id=2,
            start_ms=1000,
            end_ms=1800,
            audio_bytes=b"\x00\x01" * 1600,
            provisional_text="本句首遍错字乙",
        )
        self.assertEqual(len(self.queue.submitted), 1)
        job = self.queue.submitted[0]
        self.assertEqual(job["context"], "正确的历史句甲。")
        self.assertEqual(job["hotwords"], ["星巴克"])
        self.assertNotIn("本句首遍", job["context"] or "")

    async def test_first_sentence_submits_without_context(self):
        self.session.sentences[1] = SentenceRecord(
            sentence_id=1, start_ms=0, end_ms=800, provisional_text="欢迎使用"
        )
        await self.pipeline._process_final_segment(
            sentence_id=1,
            start_ms=0,
            end_ms=800,
            audio_bytes=b"\x00\x01" * 1600,
            provisional_text="欢迎使用",
        )
        self.assertEqual(self.queue.submitted[0]["context"], None)


class TestQwenPromptContextLimit(unittest.TestCase):
    def test_truncates_to_configured_history_chars(self):
        engine = Qwen3ASREngine(FinalASRConfig(context_history_chars=200))
        prompt = engine._build_prompt(context="汉" * 300)
        self.assertIn("(前文: " + "汉" * 200 + ")", prompt)
        self.assertNotIn("汉" * 201, prompt)
        self.assertNotIn("(上下文:", prompt)
        self.assertNotIn("前面的历史内容", prompt)

    def test_hotwords_independent_of_context(self):
        engine = Qwen3ASREngine(FinalASRConfig(context_history_chars=200))
        prompt = engine._build_prompt(hotwords=["星巴克"], context="上一句。")
        self.assertIn("(热词: 星巴克)", prompt)
        self.assertNotIn("提示词", prompt)
        self.assertIn("(前文: 上一句。)", prompt)

    def test_zero_limit_omits_context(self):
        engine = Qwen3ASREngine(FinalASRConfig(context_history_chars=0))
        prompt = engine._build_prompt(context="历史句。")
        self.assertEqual(prompt, "语音转写：")
        self.assertNotIn("前文", prompt)


if __name__ == "__main__":
    unittest.main()


class TestPromptEchoCleaning(unittest.TestCase):
    """clean_qwen_asr_text 须剥离偶发回显的 prompt 片段, 防止其进入转写正文。"""

    def test_full_prompt_echo(self):
        from core.final_asr.qwen_engine import clean_qwen_asr_text

        self.assertEqual(
            clean_qwen_asr_text("语音转写： (前文: 你订单有单号吗？有的，我订单号给你。)"),
            "",
        )

    def test_prefix_echo_before_transcript(self):
        from core.final_asr.qwen_engine import clean_qwen_asr_text

        self.assertEqual(
            clean_qwen_asr_text("语音转写：今天下午两点开会"),
            "今天下午两点开会",
        )

    def test_legacy_wording_echo(self):
        from core.final_asr.qwen_engine import clean_qwen_asr_text

        self.assertEqual(
            clean_qwen_asr_text("语音转写： (上下文: 上句内容) (提示词: 星巴克) 正文本体"),
            "正文本体",
        )

    def test_hotword_echo_with_actual_text(self):
        from core.final_asr.qwen_engine import clean_qwen_asr_text

        self.assertEqual(
            clean_qwen_asr_text("(前文: 昨天去了公园) 然后我们去看了场电影。"),
            "然后我们去看了场电影。",
        )

    def test_normal_transcript_untouched(self):
        from core.final_asr.qwen_engine import clean_qwen_asr_text

        text = "请记录：语音转写这个功能，测试一下。"
        self.assertEqual(clean_qwen_asr_text(text), text)
