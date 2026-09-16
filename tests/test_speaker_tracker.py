import unittest
import numpy as np
from core.speaker.speaker_tracker import IncrementalSpeakerTracker
from core.speaker.recluster import global_recluster_speakers


class TestSpeakerTracking(unittest.TestCase):
    def test_incremental_tracking(self):
        tracker = IncrementalSpeakerTracker(similarity_threshold=0.65, expected_speakers=2)

        # Create two distinct unit vectors (orthogonal)
        spk1_base = np.zeros(192, dtype=np.float32)
        spk1_base[0] = 1.0

        spk2_base = np.zeros(192, dtype=np.float32)
        spk2_base[1] = 1.0

        # Utterance 1 -> SPK1
        res1 = tracker.classify_and_update(spk1_base)
        self.assertEqual(res1, "SPK1")

        # Utterance 2 -> SPK1 (similar to SPK1 with small perturbation)
        spk1_variant = spk1_base.copy()
        spk1_variant[2] = 0.1
        spk1_variant = spk1_variant / np.linalg.norm(spk1_variant)
        res2 = tracker.classify_and_update(spk1_variant)
        self.assertEqual(res2, "SPK1")

        # Utterance 3 -> SPK2 (orthogonal)
        res3 = tracker.classify_and_update(spk2_base)
        self.assertEqual(res3, "SPK2")

        self.assertEqual(tracker.speaker_count, 2)

    def test_expected_speakers_constraint(self):
        tracker = IncrementalSpeakerTracker(similarity_threshold=0.80, expected_speakers=2)

        v1 = np.zeros(10, dtype=np.float32)
        v1[0] = 1.0

        v2 = np.zeros(10, dtype=np.float32)
        v2[1] = 1.0

        v3 = np.zeros(10, dtype=np.float32)
        v3[2] = 1.0

        res1 = tracker.classify_and_update(v1)  # SPK1
        res2 = tracker.classify_and_update(v2)  # SPK2
        # v3 should be mapped to one of the 2 speakers because max_speakers=expected_speakers=2
        res3 = tracker.classify_and_update(v3)
        self.assertIn(res3, ["SPK1", "SPK2"])
        self.assertEqual(tracker.speaker_count, 2)

    def test_global_reclustering(self):
        v1 = np.zeros(10, dtype=np.float32)
        v1[0] = 1.0

        v2 = np.zeros(10, dtype=np.float32)
        v2[1] = 1.0

        records = [
            {"sentence_id": 1, "speaker_embedding": v1},
            {"sentence_id": 2, "speaker_embedding": v2},
            {"sentence_id": 3, "speaker_embedding": v1 + 0.05},
            {"sentence_id": 4, "speaker_embedding": v2 + 0.05},
        ]

        speaker_map = global_recluster_speakers(records, expected_speakers=2)
        self.assertEqual(len(speaker_map), 4)
        # Sentence 1 and 3 should share the same speaker
        self.assertEqual(speaker_map[1], speaker_map[3])
        # Sentence 2 and 4 should share the same speaker
        self.assertEqual(speaker_map[2], speaker_map[4])
        # Sentence 1 and 2 should have different speakers
        self.assertNotEqual(speaker_map[1], speaker_map[2])


if __name__ == "__main__":
    unittest.main()
