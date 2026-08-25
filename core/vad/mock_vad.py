import math
import struct
from typing import List, Tuple, Dict, Any

from core.vad.base import BaseVADEngine
from config.settings import VADConfig


class MockVADEngine(BaseVADEngine):
    """
    High-fidelity Mock VAD engine for offline testing and development.
    Uses RMS energy detection with hysteresis state machine to simulate
    real FSMN-VAD speech start and end events.
    """

    def __init__(self, config: VADConfig, energy_threshold: float = 300.0):
        self.config = config
        self.energy_threshold = energy_threshold

    def _calc_rms(self, audio_bytes: bytes) -> float:
        if not audio_bytes or len(audio_bytes) < 2:
            return 0.0
        # Audio is PCM16 little-endian
        sample_count = len(audio_bytes) // 2
        try:
            samples = struct.unpack(f"<{sample_count}h", audio_bytes[: sample_count * 2])
            sum_sq = sum(s * s for s in samples)
            return math.sqrt(sum_sq / sample_count)
        except Exception:
            return 0.0

    def process_chunk(
        self, audio_bytes: bytes, cache: Dict[str, Any], is_final: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Stateful energy-based VAD simulation.
        Maintains timeline and state in cache. ``current_time_ms`` lives in
        cache, so an empty cache restarts the clock at 0 (same contract as
        FunASR FSMN-VAD).
        """
        current_time_ms = cache.get("current_time_ms", 0)
        in_speech = cache.get("in_speech", False)
        speech_start_ms = cache.get("speech_start_ms", -1)
        silence_duration_ms = cache.get("silence_duration_ms", 0)
        speech_duration_ms = cache.get("speech_duration_ms", 0)

        chunk_ms = (len(audio_bytes) // 32) if audio_bytes else 0  # 16kHz mono PCM16 = 32 bytes/ms
        current_time_ms += chunk_ms
        cache["current_time_ms"] = current_time_ms

        rms = self._calc_rms(audio_bytes)
        is_speech_frame = rms > self.energy_threshold

        results: List[Tuple[int, int]] = []

        if not in_speech:
            if is_speech_frame:
                speech_duration_ms += chunk_ms
                if speech_duration_ms >= 120:  # Need 120ms of sound to trigger speech start
                    in_speech = True
                    speech_start_ms = max(0, current_time_ms - speech_duration_ms)
                    silence_duration_ms = 0
                    results.append((speech_start_ms, -1))
            else:
                speech_duration_ms = 0
        else:
            if not is_speech_frame or is_final:
                silence_duration_ms += chunk_ms
                if silence_duration_ms >= self.config.max_end_silence_time or is_final:
                    in_speech = False
                    speech_end_ms = current_time_ms - silence_duration_ms if not is_final else current_time_ms
                    results.append((-1, speech_end_ms))
                    speech_start_ms = -1
                    speech_duration_ms = 0
                    silence_duration_ms = 0
            else:
                silence_duration_ms = 0
                speech_duration_ms += chunk_ms

        cache["in_speech"] = in_speech
        cache["speech_start_ms"] = speech_start_ms
        cache["silence_duration_ms"] = silence_duration_ms
        cache["speech_duration_ms"] = speech_duration_ms

        return results
