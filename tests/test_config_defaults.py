import os
import unittest

from config.settings import AppConfig


class TestDefaultConfig(unittest.TestCase):
    """Guard the data-driven default values (see docs/benchmark_report_*.md)."""

    def test_default_yaml_values(self):
        # Picks up config/config.default.yaml from the repo
        cfg = AppConfig.load()
        self.assertEqual(cfg.final_asr.hard_timeout_sec, 8.0)
        self.assertEqual(cfg.final_asr.max_concurrency, 32)
        self.assertEqual(cfg.final_asr.max_tokens, 512)
        self.assertEqual(cfg.final_asr.context_history_chars, 200)
        self.assertEqual(
            cfg.final_asr.vllm_url,
            "http://127.0.0.1:8899/v1/audio/transcriptions",
        )
        self.assertEqual(cfg.final_asr.model_name, "qwen3-asr")
        self.assertEqual(cfg.server.max_connections, 100)
        self.assertEqual(cfg.server.max_audio_frame_ms, 2000)
        self.assertTrue(cfg.admission.enable)
        self.assertEqual(cfg.admission.streaming_wait_ms_limit, 200.0)
        self.assertTrue(cfg.voice.suppress_prestart_partials)
        self.assertEqual(cfg.voice.silence_floor_dbfs, -80.0)
        self.assertFalse(cfg.voice.skip_inference_on_silence)
        self.assertTrue(cfg.voice.watchdog_enable)
        self.assertEqual(cfg.voice.watchdog_min_chars, 8)
        self.assertEqual(cfg.voice.watchdog_max_segment_ms, 30000)
        self.assertEqual(cfg.voice.watchdog_rate_limit_per_hour, 20)

    def test_worker_threads_auto_resolves_to_cpu_bound(self):
        cfg = AppConfig()
        self.assertEqual(cfg.pool.worker_threads, 0)  # 0 = auto sentinel
        cfg.resolve_devices()
        self.assertEqual(cfg.pool.worker_threads, min(16, os.cpu_count() or 4))

    def test_worker_threads_explicit_value_preserved(self):
        cfg = AppConfig()
        cfg.pool.worker_threads = 7
        cfg.resolve_devices()
        self.assertEqual(cfg.pool.worker_threads, 7)


    def test_final_asr_dataclass_defaults_align_readme(self):
        from config.settings import FinalASRConfig

        cfg = FinalASRConfig()
        self.assertEqual(
            cfg.vllm_url, "http://127.0.0.1:8899/v1/audio/transcriptions"
        )
        self.assertEqual(cfg.model_name, "qwen3-asr")


class TestTuningEnvOverrides(unittest.TestCase):

    """New tuning knobs must be reachable from the environment (compose)."""

    _VARS = (
        "MAX_CONNECTIONS",
        "ASR_WORKER_THREADS",
        "FINAL_TIMEOUT_SEC",
        "FINAL_MAX_CONCURRENCY",
        "FINAL_MAX_TOKENS",
        "FINAL_CONTEXT_HISTORY_CHARS",
        "VOICE_SUPPRESS_PRESTART_PARTIALS",
        "VOICE_WATCHDOG_ENABLE",
        "VOICE_SILENCE_FLOOR_DBFS",
        "VOICE_WATCHDOG_MIN_CHARS",
        "VOICE_SKIP_INFERENCE_ON_SILENCE",
    )

    def setUp(self):
        self._saved = {v: os.environ.get(v) for v in self._VARS}
        for v in self._VARS:
            os.environ.pop(v, None)

    def tearDown(self):
        for v, val in self._saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

    def test_env_overrides_applied(self):
        os.environ["MAX_CONNECTIONS"] = "32"
        os.environ["ASR_WORKER_THREADS"] = "4"
        os.environ["FINAL_TIMEOUT_SEC"] = "5"
        os.environ["FINAL_MAX_CONCURRENCY"] = "24"
        os.environ["FINAL_MAX_TOKENS"] = "384"
        os.environ["FINAL_CONTEXT_HISTORY_CHARS"] = "120"
        os.environ["VOICE_SUPPRESS_PRESTART_PARTIALS"] = "false"
        os.environ["VOICE_WATCHDOG_ENABLE"] = "false"
        os.environ["VOICE_SILENCE_FLOOR_DBFS"] = "-75"
        os.environ["VOICE_WATCHDOG_MIN_CHARS"] = "12"
        os.environ["VOICE_SKIP_INFERENCE_ON_SILENCE"] = "1"
        cfg = AppConfig.load()
        self.assertEqual(cfg.server.max_connections, 32)
        self.assertEqual(cfg.pool.worker_threads, 4)
        self.assertEqual(cfg.final_asr.hard_timeout_sec, 5.0)
        self.assertEqual(cfg.final_asr.max_concurrency, 24)
        self.assertEqual(cfg.final_asr.max_tokens, 384)
        self.assertEqual(cfg.final_asr.context_history_chars, 120)
        self.assertFalse(cfg.voice.suppress_prestart_partials)
        self.assertFalse(cfg.voice.watchdog_enable)
        self.assertEqual(cfg.voice.silence_floor_dbfs, -75.0)
        self.assertEqual(cfg.voice.watchdog_min_chars, 12)
        self.assertTrue(cfg.voice.skip_inference_on_silence)

    def test_no_env_uses_defaults(self):
        cfg = AppConfig.load()
        self.assertEqual(cfg.server.max_connections, 100)
        self.assertEqual(cfg.final_asr.hard_timeout_sec, 8.0)
        # worker_threads resolved to the auto value, not left at the sentinel
        self.assertGreater(cfg.pool.worker_threads, 0)
        self.assertTrue(cfg.voice.suppress_prestart_partials)
        self.assertTrue(cfg.voice.watchdog_enable)
        self.assertEqual(cfg.voice.watchdog_min_chars, 8)
        self.assertEqual(cfg.voice.watchdog_max_segment_ms, 30000)
        self.assertEqual(cfg.voice.watchdog_rate_limit_per_hour, 20)

    def test_watchdog_segment_must_fit_ring_buffer(self):
        cfg = AppConfig.load()
        cfg.voice.watchdog_max_segment_ms = int(
            cfg.audio.ring_buffer_duration_sec * 1000
        )  # 32s ring + 0.8s preroll exceeds budget
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
