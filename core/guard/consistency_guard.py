import re
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

from config.settings import GuardConfig


@dataclass
class GuardDecision:
    is_valid: bool
    needs_review: bool
    revision_distance: float
    final_source: str  # "qwen3-asr" | "paraformer-fallback"
    selected_text: str
    rejection_reason: Optional[str] = None


def remove_punctuation(text: str) -> str:
    """Strip all Chinese and English punctuation marks and whitespace for normalized comparison."""
    if not text:
        return ""
    punc_pattern = r"[^\w\u4e00-\u9fa5]"
    return re.sub(punc_pattern, "", text).strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]


def compute_normalized_edit_distance(text1: str, text2: str) -> float:
    """
    Compute edit distance normalized by max text length.
    Returns float in [0.0, 1.0].
    """
    clean1 = remove_punctuation(text1)
    clean2 = remove_punctuation(text2)

    max_len = max(len(clean1), len(clean2))
    if max_len == 0:
        return 0.0

    dist = levenshtein_distance(clean1, clean2)
    return dist / max_len


def check_ngram_repetition(text: str, ngram: int = 4, max_repeat: int = 3) -> bool:
    """
    Check if text contains excessive repeated n-grams (hallucination loop).
    """
    clean = remove_punctuation(text)
    if len(clean) < ngram * max_repeat:
        return False

    for n in range(2, ngram + 1):
        for i in range(len(clean) - n + 1):
            pattern = clean[i : i + n]
            # Check consecutive repetitions
            repeated_pattern = pattern * max_repeat
            if repeated_pattern in clean:
                return True
    return False


class ConsistencyGuard:
    """
    Guards ASR results against LLM hallucination, context drift, and abnormal insertions.
    Flags suspicious Qwen3-ASR finals for human review (needs_review=True) but keeps
    the Qwen text as the committed result.

    Policy: the first pass (streaming Paraformer) is measurably worse than the
    second pass, so replacing an anomalous Qwen output with the provisional text
    almost always degrades quality. Anomalies (hallucination-style rates,
    repetition loops, abnormal insertions) are therefore flagged, not swapped.
    An empty Qwen output from a successful response is committed as-is
    (authoritative "no transcribable speech" verdict). Fallback to provisional
    text happens ONLY in the pipeline's exception paths: timeout, inference
    failure, or queue overflow, where no Qwen output exists at all.
    """

    def __init__(self, config: GuardConfig):
        self.config = config

    def evaluate(
        self,
        provisional_text: str,
        qwen_text: str,
        duration_ms: int,
    ) -> GuardDecision:
        """
        Evaluate Qwen Final output against Paraformer Provisional output and audio duration.
        """
        if not self.config.enable:
            return GuardDecision(
                is_valid=True,
                needs_review=False,
                revision_distance=0.0,
                final_source="qwen3-asr",
                selected_text=qwen_text if qwen_text else provisional_text,
            )

        clean_prov = remove_punctuation(provisional_text)
        clean_qwen = remove_punctuation(qwen_text)

        # 1. Empty Qwen output on a successful response is an authoritative
        #    verdict (the segment contains no transcribable speech): commit it
        #    as-is. Fallback happens only on explicit timeout / failure /
        #    unavailability, which the pipeline handles via exceptions.
        if not clean_qwen:
            logger.debug(
                f"[Guard] Qwen returned empty output for segment, committing empty final "
                f"(provisional was '{provisional_text}')"
            )
            return GuardDecision(
                is_valid=True,
                needs_review=False,
                revision_distance=1.0 if clean_prov else 0.0,
                final_source="qwen3-asr",
                selected_text="",
            )

        # 2. Audio duration vs text length check (Hallucination rate test)
        duration_sec = max(0.1, duration_ms / 1000.0)
        chars_per_sec = len(clean_qwen) / duration_sec
        if duration_ms < self.config.min_speech_duration_ms and len(clean_qwen) > 10:
            logger.warning(
                f"[Guard] Audio too short ({duration_ms}ms) for text length ({len(clean_qwen)} chars): '{qwen_text}'"
            )
            return GuardDecision(
                is_valid=False,
                needs_review=True,
                revision_distance=1.0,
                final_source="qwen3-asr",
                selected_text=qwen_text,
                rejection_reason=f"Suspicious text length for {duration_ms}ms audio",
            )

        if chars_per_sec > self.config.max_chars_per_second:
            logger.warning(
                f"[Guard] Speech speed ({chars_per_sec:.1f} chars/s) exceeds threshold: '{qwen_text}'"
            )
            return GuardDecision(
                is_valid=False,
                needs_review=True,
                revision_distance=1.0,
                final_source="qwen3-asr",
                selected_text=qwen_text,
                rejection_reason=f"Abnormal speech rate ({chars_per_sec:.1f} chars/s)",
            )

        # 3. Repetition loop check
        if check_ngram_repetition(
            clean_qwen,
            ngram=self.config.repetition_ngram,
            max_repeat=self.config.repetition_max_count,
        ):
            logger.warning(f"[Guard] Repetition loop detected in Qwen output: '{qwen_text}'")
            return GuardDecision(
                is_valid=False,
                needs_review=True,
                revision_distance=1.0,
                final_source="qwen3-asr",
                selected_text=qwen_text,
                rejection_reason="Repetition loop detected",
            )

        # 4. If Paraformer is empty (<2 chars) but Qwen hallucinated a long sentence (>12 chars)
        if len(clean_prov) <= 1 and len(clean_qwen) > 12:
            logger.warning(
                f"[Guard] Abnormal insertion: Paraformer empty but Qwen output '{qwen_text}'"
            )
            return GuardDecision(
                is_valid=False,
                needs_review=True,
                revision_distance=1.0,
                final_source="qwen3-asr",
                selected_text=qwen_text,
                rejection_reason="Abnormal insertion without Paraformer agreement",
            )

        # 5. Normalized edit distance check
        distance = compute_normalized_edit_distance(clean_prov, clean_qwen)

        if distance > self.config.max_edit_distance_ratio:
            # Significant divergence between models
            logger.warning(
                f"[Guard] Large revision distance ({distance:.2f} > {self.config.max_edit_distance_ratio}): "
                f"Prov='{provisional_text}' vs Qwen='{qwen_text}'"
            )
            return GuardDecision(
                is_valid=True,
                needs_review=True,
                revision_distance=distance,
                final_source="qwen3-asr",
                selected_text=qwen_text,
                rejection_reason="Large revision distance",
            )

        # Normal valid revision
        return GuardDecision(
            is_valid=True,
            needs_review=False,
            revision_distance=distance,
            final_source="qwen3-asr",
            selected_text=qwen_text,
        )
