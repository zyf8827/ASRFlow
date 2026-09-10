import unittest
from config.settings import GuardConfig
from core.guard.consistency_guard import (
    ConsistencyGuard,
    compute_normalized_edit_distance,
    check_ngram_repetition,
    remove_punctuation,
)


class TestConsistencyGuard(unittest.TestCase):
    def setUp(self):
        self.config = GuardConfig(
            enable=True,
            max_edit_distance_ratio=0.60,
            max_chars_per_second=12.0,
            min_speech_duration_ms=300,
            repetition_ngram=4,
            repetition_max_count=3,
        )
        self.guard = ConsistencyGuard(self.config)

    def test_remove_punctuation(self):
        text = "你好，世界！今天天气不错... 是吗？"
        clean = remove_punctuation(text)
        self.assertEqual(clean, "你好世界今天天气不错是吗")

    def test_normalized_edit_distance(self):
        s1 = "我昨天下午去了公园"
        s2 = "我昨天下午去了城市公园。"
        dist = compute_normalized_edit_distance(s1, s2)
        self.assertLess(dist, 0.3)

        s3 = "今天天气很好"
        s4 = "明天上午开会讨论方案"
        dist2 = compute_normalized_edit_distance(s3, s4)
        self.assertGreater(dist2, 0.7)

    def test_repetition_detection(self):
        text1 = "对对对对对对对对"
        self.assertTrue(check_ngram_repetition(text1, ngram=2, max_repeat=3))

        text2 = "你好世界，很高兴认识你。"
        self.assertFalse(check_ngram_repetition(text2, ngram=4, max_repeat=3))

        text3 = "谢谢大家谢谢大家谢谢大家"
        self.assertTrue(check_ngram_repetition(text3, ngram=4, max_repeat=3))

    def test_guard_valid_revision(self):
        decision = self.guard.evaluate(
            provisional_text="我昨天下午去了公园",
            qwen_text="我昨天下午去了城市公园。",
            duration_ms=3000,
        )
        self.assertTrue(decision.is_valid)
        self.assertFalse(decision.needs_review)
        self.assertEqual(decision.final_source, "qwen3-asr")
        self.assertEqual(decision.selected_text, "我昨天下午去了城市公园。")

    def test_guard_repetition_hallucination(self):
        # 首遍质量显著差于二遍: 异常输出保留 qwen 文本并标记复核, 不再回退首遍
        decision = self.guard.evaluate(
            provisional_text="谢谢大家",
            qwen_text="谢谢大家谢谢大家谢谢大家谢谢大家。",
            duration_ms=1000,
        )
        self.assertFalse(decision.is_valid)
        self.assertTrue(decision.needs_review)
        self.assertEqual(decision.final_source, "qwen3-asr")
        self.assertEqual(decision.selected_text, "谢谢大家谢谢大家谢谢大家谢谢大家。")

    def test_guard_duration_too_short(self):
        # 200ms audio yielding 20 Chinese characters -> hallucination flagged,
        # qwen text kept for review instead of falling back to the worse first pass
        decision = self.guard.evaluate(
            provisional_text="好",
            qwen_text="欢迎光临星巴克请问您今天想喝什么口味的咖啡。",
            duration_ms=200,
        )
        self.assertFalse(decision.is_valid)
        self.assertTrue(decision.needs_review)
        self.assertEqual(decision.final_source, "qwen3-asr")
        self.assertEqual(decision.selected_text, "欢迎光临星巴克请问您今天想喝什么口味的咖啡。")

    def test_guard_empty_qwen_output_is_authoritative(self):
        # 成功响应但输出为空 = offline 判定该段无有效语音, 以 offline 为准定稿空文本,
        # 不回退首遍 (回退只保留给超时/失败/不可用, 由管线异常分支处理)
        decision = self.guard.evaluate(
            provisional_text="不",
            qwen_text="",
            duration_ms=200,
        )
        self.assertTrue(decision.is_valid)
        self.assertFalse(decision.needs_review)
        self.assertEqual(decision.final_source, "qwen3-asr")
        self.assertEqual(decision.selected_text, "")

        # 输出仅含标点/空白, 清洗后为空, 同样视为空判定
        decision2 = self.guard.evaluate(
            provisional_text="嗯",
            qwen_text="。",
            duration_ms=150,
        )
        self.assertTrue(decision2.is_valid)
        self.assertEqual(decision2.final_source, "qwen3-asr")
        self.assertEqual(decision2.selected_text, "")

        # 双空: 距离为 0
        decision3 = self.guard.evaluate(
            provisional_text="",
            qwen_text="",
            duration_ms=100,
        )
        self.assertTrue(decision3.is_valid)
        self.assertEqual(decision3.final_source, "qwen3-asr")
        self.assertEqual(decision3.selected_text, "")
        self.assertEqual(decision3.revision_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
