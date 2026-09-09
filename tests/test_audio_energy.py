"""Unit tests for the PCM16 RMS helper used by L0."""
import math
import random
import struct
import unittest

from core.audio_energy import DIGITAL_SILENCE_DBFS, pcm16_rms_dbfs


def _pcm(samples):
    return struct.pack(f"<{len(samples)}h", *samples)


class TestPcm16RmsDbfs(unittest.TestCase):
    def test_digital_silence_is_sentinel(self):
        self.assertEqual(pcm16_rms_dbfs(b""), DIGITAL_SILENCE_DBFS)
        self.assertEqual(pcm16_rms_dbfs(b"\x00"), DIGITAL_SILENCE_DBFS)  # half sample
        self.assertEqual(pcm16_rms_dbfs(_pcm([0] * 16000)), DIGITAL_SILENCE_DBFS)

    def test_tone_is_well_above_silence_floor(self):
        # amp 8000/32768 ≈ -12 dBFS
        samples = [int(8000 * math.sin(2 * math.pi * 260 * i / 16000)) for i in range(16000)]
        dbfs = pcm16_rms_dbfs(_pcm(samples))
        self.assertGreater(dbfs, -20.0)
        self.assertLess(dbfs, -8.0)

    def test_quiet_analog_noise_between_l0_and_speech(self):
        # amp ±10 uniform noise ≈ -75 dBFS: above digital silence and above
        # the -80 L0 gate, far below speech level
        random.seed(7)
        samples = [random.randint(-10, 10) for _ in range(16000)]
        dbfs = pcm16_rms_dbfs(_pcm(samples))
        self.assertGreater(dbfs, DIGITAL_SILENCE_DBFS)
        self.assertGreater(dbfs, -80.0)
        self.assertLess(dbfs, -60.0)


if __name__ == "__main__":
    unittest.main()
