import os
import re
from typing import List, Dict, Optional, Tuple


class HotwordManager:
    """
    Manages session hotwords and domain terminology.
    - format_paraformer_hotwords: cleans/dedupes the session list (API reserved;
      current Pass-1 ONNX engine ignores hotwords — no decode bias).
    - Generates concise prompt hints for Qwen3-ASR without triggering hallucination.
    - Performs deterministic post-processing dictionary replacement (e.g. "wrong=>right" pairs).
    """

    def __init__(self, global_dictionary: Optional[Dict[str, str]] = None):
        # Default is empty; domain-specific mappings should be loaded via config/file or POSTPROCESS_HOTWORDS
        if global_dictionary is None:
            self.global_dictionary: Dict[str, str] = {}
        else:
            self.global_dictionary = dict(global_dictionary)

    def format_paraformer_hotwords(self, session_hotwords: List[str]) -> List[str]:
        """Deduplicate and clean hotwords (Pass-1 ONNX currently ignores these)."""
        seen = set()
        cleaned = []
        for hw in session_hotwords:
            w = hw.strip()
            if w and w not in seen:
                seen.add(w)
                cleaned.append(w)
        return cleaned

    def format_qwen_context(self, session_hotwords: List[str], max_words: int = 5) -> str:
        """
        Generate a concise hint for Qwen prompt.
        Limits to top N words to avoid context distraction / hallucination.
        """
        if not session_hotwords:
            return ""
        return ", ".join(session_hotwords[:max_words])

    @staticmethod
    def parse_postprocess_hotwords_payload(payload: str) -> Dict[str, str]:
        """
        Parse postprocess hotwords in format:
        "错别字=>正确字, 拼音词=>专有名词" or newline delimited.
        """
        result = {}
        if not payload:
            return result

        # Split by comma or newline
        lines = re.split(r"[\n,]", payload)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "=>" in line:
                parts = line.split("=>", 1)
                src, dst = parts[0].strip(), parts[1].strip()
                if src and dst:
                    result[src] = dst
            elif "->" in line:
                parts = line.split("->", 1)
                src, dst = parts[0].strip(), parts[1].strip()
                if src and dst:
                    result[src] = dst
        return result

    @classmethod
    def load_hotword_file(cls, file_path: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Load hotwords file supporting both plain word list and 'wrong=>right' mappings.
        """
        hotwords = []
        postprocess_dict = {}
        if not os.path.exists(file_path):
            return hotwords, postprocess_dict

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=>" in line or "->" in line:
                    sep = "=>" if "=>" in line else "->"
                    src, dst = line.split(sep, 1)
                    postprocess_dict[src.strip()] = dst.strip()
                else:
                    hotwords.append(line)
        return hotwords, postprocess_dict

    def postprocess_replace(self, text: str, custom_dict: Optional[Dict[str, str]] = None) -> str:
        """
        Deterministic post-recognition dictionary correction.
        """
        if not text:
            return ""
        result = text
        merged_dict = dict(self.global_dictionary)
        if custom_dict:
            merged_dict.update(custom_dict)

        for typo, correct in merged_dict.items():
            if typo in result:
                result = result.replace(typo, correct)
        return result
