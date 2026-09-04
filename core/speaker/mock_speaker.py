import math
import struct
from typing import Optional
import numpy as np

from core.speaker.base import BaseSpeakerEngine
from config.settings import SpeakerConfig


class MockSpeakerEngine(BaseSpeakerEngine):
    """
    Mock speaker embedding engine for testing.
    Generates deterministic, normalized 192-dim vectors.
    """

    def __init__(self, config: SpeakerConfig, dim: int = 192):
        self.config = config
        self.dim = dim

    def extract_embedding(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        if not audio_bytes or len(audio_bytes) < 3200:
            return None

        # Derive a pseudo-speaker signature from average amplitude and byte variance
        sample_count = min(1000, len(audio_bytes) // 2)
        samples = struct.unpack(f"<{sample_count}h", audio_bytes[: sample_count * 2])
        mean_val = sum(samples) / sample_count
        variance = sum((s - mean_val) ** 2 for s in samples) / sample_count
        base_seed = int(variance) % 5  # Simulates ~5 different voices

        # Generate base vector for this speaker seed
        np.random.seed(base_seed + 42)
        base_vec = np.random.randn(self.dim).astype(np.float32)

        # Add small random jitter to simulate within-speaker variation
        jitter = np.random.randn(self.dim).astype(np.float32) * 0.05
        vec = base_vec + jitter
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec
