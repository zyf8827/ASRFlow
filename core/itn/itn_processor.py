import re
from typing import Optional, Union
from loguru import logger

from config.settings import ITNConfig


CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

CN_UNIT_MAP = {
    "十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000
}


def chinese_to_number(cn_str: str) -> Optional[Union[int, str]]:
    """
    Convert a simple Chinese numeral string to integer (e.g. '三千五百' -> 3500, '十二' -> 12).
    Returns str for pure digit sequences (phone numbers) like '一三八...' -> '138...'.
    """
    if not cn_str:
        return None

    # Check if purely sequential digits like phone numbers (e.g. 一三八零零一三八零零零)
    if all(c in CN_NUM_MAP for c in cn_str) and not any(u in cn_str for u in CN_UNIT_MAP):
        return "".join(str(CN_NUM_MAP[c]) for c in cn_str)

    total = 0
    r = 1  # unit
    current_section = 0
    section_val = 0

    try:
        # Standard Chinese numeral parsing
        i = 0
        n = len(cn_str)
        num = 0
        has_num = False
        
        while i < n:
            char = cn_str[i]
            if char in CN_NUM_MAP:
                num = CN_NUM_MAP[char]
                has_num = True
                if i == n - 1:
                    section_val += num
            elif char in ("十", "百", "千"):
                unit = CN_UNIT_MAP[char]
                if not has_num:
                    num = 1  # For cases like '十二' -> 12
                section_val += num * unit
                num = 0
                has_num = False
            elif char in ("万", "亿"):
                unit = CN_UNIT_MAP[char]
                if has_num:
                    section_val += num
                    num = 0
                    has_num = False
                section_val = (section_val if section_val > 0 else 1) * unit
                total += section_val
                section_val = 0
            i += 1
        total += section_val
        return total
    except Exception:
        return None


class ITNProcessor:
    """
    Deterministic Chinese Inverse Text Normalizer.
    Converts spoken expressions (numbers, currency, dates, percentages) into standardized written form.
    """

    def __init__(self, config: ITNConfig):
        self.config = config

    def normalize(self, text: str) -> str:
        if not text or not self.config.enable:
            return text

        result = text

        try:
            # 1. Percentages: 百分之五十 -> 50%
            def replace_percentage(match):
                cn_val = match.group(1)
                num = chinese_to_number(cn_val)
                return f"{num}%" if num is not None else match.group(0)

            result = re.sub(r"百分之([零一二两三四五六七八九十百]+)", replace_percentage, result)

            # 2. Dates: 二零二六年八月二十四日 -> 2026年8月24日
            def replace_year(match):
                year_cn = match.group(1)
                digits = "".join(str(CN_NUM_MAP.get(c, c)) for c in year_cn)
                return f"{digits}年"

            result = re.sub(r"([零一二三四五六七八九]{2,4})年", replace_year, result)

            def replace_month_day(match):
                m_cn = match.group(1)
                d_cn = match.group(2)
                m = chinese_to_number(m_cn)
                d = chinese_to_number(d_cn)
                m_str = str(m) if m is not None else m_cn
                d_str = str(d) if d is not None else d_cn
                return f"{m_str}月{d_str}日"

            result = re.sub(r"([一二三四五六七八九十]+)月([一二三四五六七八九十]+)[日号]", replace_month_day, result)

            # 3. Currency / Quantities: 三千元 -> 3000元, 五十块钱 -> 50块钱
            def replace_money(match):
                cn_num = match.group(1)
                unit = match.group(2)
                num = chinese_to_number(cn_num)
                return f"{num}{unit}" if num is not None else match.group(0)

            result = re.sub(
                r"([零一二两三四五六七八九十百千万亿]+)(元|块钱|万元|亿元|美元|欧元)",
                replace_money,
                result,
            )

            # 4. Long phone / serial numbers (>=5 consecutive digits)
            def replace_serial_numbers(match):
                cn_digits = match.group(0)
                return "".join(str(CN_NUM_MAP[c]) for c in cn_digits)

            result = re.sub(r"[零一二三四五六七八九]{5,}", replace_serial_numbers, result)

            # 5. General standalone number phrases with units (e.g. 第十二个 -> 第12个)
            def replace_ordinal(match):
                cn_num = match.group(1)
                num = chinese_to_number(cn_num)
                return f"第{num}" if num is not None else match.group(0)

            result = re.sub(r"第([一二三四五六七八九十百千]+)", replace_ordinal, result)

        except Exception as e:
            logger.warning(f"[ITNProcessor] Error normalizing text '{text}': {e}")
            return text

        return result
