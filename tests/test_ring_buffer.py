import unittest
from core.ring_buffer import AudioRingBuffer


class TestAudioRingBuffer(unittest.TestCase):
    def setUp(self):
        # 16kHz, mono, PCM16 -> 32 bytes per ms
        self.rb = AudioRingBuffer(
            sample_rate=16000,
            channels=1,
            bytes_per_sample=2,
            max_duration_sec=2.0,  # 2000ms = 64000 bytes
            pre_roll_ms=500,       # 500ms = 16000 bytes
        )

    def test_write_and_timeline(self):
        # 100ms chunk = 3200 bytes
        chunk = b"\x01\x02" * 1600
        ms = self.rb.write(chunk)
        self.assertEqual(ms, 100)
        self.assertEqual(self.rb.total_ms, 100)

        ms = self.rb.write(chunk)
        self.assertEqual(ms, 200)
        self.assertEqual(self.rb.total_ms, 200)

    def test_get_range(self):
        # Write 500ms
        chunk = b"A" * 3200
        for _ in range(5):
            self.rb.write(chunk)
        self.assertEqual(self.rb.total_ms, 500)

        # Slice 100ms ~ 300ms (200ms = 6400 bytes)
        sliced = self.rb.get_range(100, 300)
        self.assertEqual(len(sliced), 6400)
        self.assertTrue(all(b == ord("A") for b in sliced))

    def test_preroll_extraction(self):
        # Write 1000ms of data
        chunk = b"B" * 3200
        for _ in range(10):
            self.rb.write(chunk)

        # VAD speech from 600ms to 900ms, with pre-roll 500ms -> should extract from 100ms to 900ms (800ms = 25600 bytes)
        seg = self.rb.get_segment_with_preroll(start_ms=600, end_ms=900, custom_preroll_ms=500)
        self.assertEqual(len(seg), 800 * 32)

    def test_circular_wraparound(self):
        # Buffer max is 2000ms. Write 3000ms (96000 bytes)
        chunk = b"C" * 3200
        for _ in range(30):
            self.rb.write(chunk)

        self.assertEqual(self.rb.total_ms, 3000)
        # Slicing recent range [2500, 2800] (300ms)
        seg = self.rb.get_range(2500, 2800)
        self.assertEqual(len(seg), 300 * 32)

        # Requesting data older than max_duration should be clamped
        seg_old = self.rb.get_range(0, 500)
        self.assertEqual(len(seg_old), 0)


if __name__ == "__main__":
    unittest.main()
