import unittest
from config.settings import ITNConfig
from core.itn.itn_processor import ITNProcessor, chinese_to_number


class TestITNProcessor(unittest.TestCase):
    def setUp(self):
        self.config = ITNConfig(enable=True)
        self.itn = ITNProcessor(self.config)

    def test_chinese_to_number(self):
        self.assertEqual(chinese_to_number("十二"), 12)
        self.assertEqual(chinese_to_number("三千五百"), 3500)
        self.assertEqual(chinese_to_number("一万零八"), 10008)
        self.assertEqual(chinese_to_number("一三八零零"), "13800")

    def test_percentage_normalization(self):
        text = "本次达成率达到了百分之八十五。"
        norm = self.itn.normalize(text)
        self.assertEqual(norm, "本次达成率达到了85%。")

    def test_date_normalization(self):
        text = "会议定在二零二六年八月二十四日举行。"
        norm = self.itn.normalize(text)
        self.assertEqual(norm, "会议定在2026年8月24日举行。")

    def test_money_normalization(self):
        text = "这杯咖啡一共三十元，电影票还花了五十块钱。"
        norm = self.itn.normalize(text)
        self.assertEqual(norm, "这杯咖啡一共30元，电影票还花了50块钱。")

    def test_serial_phone_numbers(self):
        text = "请拨打电话一三八零零一三八零零零联系客服。"
        norm = self.itn.normalize(text)
        self.assertEqual(norm, "请拨打电话13800138000联系客服。")


if __name__ == "__main__":
    unittest.main()
