"""Product engines must not silently fall back to Mock when backend is auto."""
import unittest
from unittest.mock import patch


class TestStreamingASRFactory(unittest.TestCase):
    def test_explicit_mock_ok(self):
        from config.settings import StreamingASRConfig
        from core.streaming_asr import create_streaming_asr_engine
        from core.streaming_asr.mock_streaming import MockStreamingASREngine

        eng = create_streaming_asr_engine(StreamingASRConfig(backend="mock"))
        self.assertIsInstance(eng, MockStreamingASREngine)

    def test_auto_refuses_silent_mock_on_init_failure(self):
        from config.settings import StreamingASRConfig
        from core.streaming_asr import create_streaming_asr_engine

        with patch(
            "core.streaming_asr.onnx_batched_streaming.OnnxBatchedStreamingEngine",
            side_effect=RuntimeError("no onnx"),
            create=True,
        ):
            # Patch the import path used inside the factory
            import core.streaming_asr.onnx_batched_streaming as mod

            with patch.object(
                mod, "OnnxBatchedStreamingEngine", side_effect=RuntimeError("no onnx")
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    create_streaming_asr_engine(
                        StreamingASRConfig(
                            backend="auto", onnx_model_dir="/nonexistent/onnx"
                        )
                    )
                self.assertIn("refusing silent Mock fallback", str(ctx.exception))

    def test_auto_missing_model_dir_raises(self):
        from config.settings import StreamingASRConfig
        from core.streaming_asr import create_streaming_asr_engine

        with self.assertRaises(RuntimeError) as ctx:
            create_streaming_asr_engine(
                StreamingASRConfig(backend="auto", onnx_model_dir="/nonexistent/onnx")
            )
        msg = str(ctx.exception)
        self.assertIn("refusing silent Mock fallback", msg)
        self.assertNotIsInstance(ctx.exception, type(None))


class TestVADFactory(unittest.TestCase):
    def test_explicit_mock_ok(self):
        from config.settings import VADConfig
        from core.vad import create_vad_engine
        from core.vad.mock_vad import MockVADEngine

        self.assertIsInstance(create_vad_engine(VADConfig(backend="mock")), MockVADEngine)

    def test_auto_refuses_silent_mock(self):
        from config.settings import VADConfig
        from core.vad import create_vad_engine

        with patch(
            "core.vad.fsmn_vad.FSMNVADEngine",
            side_effect=ImportError("no funasr"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                create_vad_engine(VADConfig(backend="auto"))
            self.assertIn("refusing silent Mock fallback", str(ctx.exception))


class TestSpeakerFactory(unittest.TestCase):
    def test_explicit_mock_ok(self):
        from config.settings import SpeakerConfig
        from core.speaker import create_speaker_engine
        from core.speaker.mock_speaker import MockSpeakerEngine

        self.assertIsInstance(
            create_speaker_engine(SpeakerConfig(backend="mock")), MockSpeakerEngine
        )

    def test_auto_refuses_silent_mock(self):
        from config.settings import SpeakerConfig
        from core.speaker import create_speaker_engine

        with patch(
            "core.speaker.eres2net_extractor.ERes2NetSpeakerEngine",
            side_effect=ImportError("no funasr"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                create_speaker_engine(SpeakerConfig(backend="auto"))
            self.assertIn("refusing silent Mock fallback", str(ctx.exception))


class TestPuncFactory(unittest.TestCase):
    def test_explicit_mock_ok(self):
        from config.settings import PuncConfig
        from core.punc import create_punc_engine
        from core.punc.mock_punc import MockPuncEngine

        self.assertIsInstance(create_punc_engine(PuncConfig(backend="mock")), MockPuncEngine)

    def test_auto_refuses_silent_mock(self):
        from config.settings import PuncConfig
        from core.punc import create_punc_engine

        with patch(
            "core.punc.ct_punc.CTPuncEngine",
            side_effect=ImportError("no funasr"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                create_punc_engine(PuncConfig(backend="auto"))
            self.assertIn("refusing silent Mock fallback", str(ctx.exception))


class TestFinalASRFactory(unittest.TestCase):
    def test_explicit_mock_ok(self):
        from config.settings import FinalASRConfig
        from core.final_asr import create_final_asr_engine
        from core.final_asr.mock_qwen import MockQwenEngine

        self.assertIsInstance(
            create_final_asr_engine(FinalASRConfig(engine_type="mock")), MockQwenEngine
        )

    def test_auto_resolves_to_qwen_not_mock(self):
        from config.settings import FinalASRConfig
        from core.final_asr import create_final_asr_engine
        from core.final_asr.qwen_engine import Qwen3ASREngine
        from core.final_asr.mock_qwen import MockQwenEngine

        eng = create_final_asr_engine(FinalASRConfig(engine_type="auto"))
        self.assertIsInstance(eng, Qwen3ASREngine)
        self.assertNotIsInstance(eng, MockQwenEngine)

    def test_vllm_http_selected(self):
        from config.settings import FinalASRConfig
        from core.final_asr import create_final_asr_engine
        from core.final_asr.qwen_engine import Qwen3ASREngine

        eng = create_final_asr_engine(FinalASRConfig(engine_type="vllm_http"))
        self.assertIsInstance(eng, Qwen3ASREngine)


if __name__ == "__main__":
    unittest.main()
