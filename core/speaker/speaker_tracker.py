from typing import Dict, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SpeakerCluster:
    speaker_id: str
    centroid: np.ndarray
    count: int = 1

    def update(self, embedding: np.ndarray):
        # Running average update
        new_centroid = self.centroid * self.count + embedding
        self.count += 1
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-6:
            self.centroid = new_centroid / norm
        else:
            self.centroid = new_centroid


class IncrementalSpeakerTracker:
    """
    Real-time incremental speaker tracker.
    Uses running mean centroids and cosine similarity thresholding.
    Supports expected_speakers constraint to suppress speaker drift.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        expected_speakers: Optional[int] = None,
        max_speakers: int = 10,
    ):
        self.similarity_threshold = similarity_threshold
        self.expected_speakers = expected_speakers
        self.max_speakers = max_speakers
        self.clusters: Dict[str, SpeakerCluster] = {}
        self._next_spk_idx = 1

    @property
    def speaker_count(self) -> int:
        return len(self.clusters)

    def classify_and_update(self, embedding: Optional[np.ndarray]) -> Optional[str]:
        """
        Classify incoming embedding vector and update cluster centroids.
        Returns speaker ID string (e.g. "SPK1", "SPK2") or None if embedding is invalid.
        """
        if embedding is None:
            return None

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None
        embedding = (embedding / norm).astype(np.float32)

        if not self.clusters:
            # First speaker
            spk_id = f"SPK{self._next_spk_idx}"
            self._next_spk_idx += 1
            self.clusters[spk_id] = SpeakerCluster(
                speaker_id=spk_id,
                centroid=embedding.copy(),
                count=1,
            )
            return spk_id

        # Compute cosine similarity with all existing centroids
        best_spk_id = None
        best_sim = -1.0

        for spk_id, cluster in self.clusters.items():
            sim = float(np.dot(embedding, cluster.centroid))
            if sim > best_sim:
                best_sim = sim
                best_spk_id = spk_id

        # Decision logic
        if best_sim >= self.similarity_threshold:
            # Match existing cluster
            self.clusters[best_spk_id].update(embedding)
            return best_spk_id

        # Does not exceed threshold: Check if we are allowed to create a new cluster
        can_create_new = True
        if self.expected_speakers is not None and len(self.clusters) >= self.expected_speakers:
            can_create_new = False
        elif len(self.clusters) >= self.max_speakers:
            can_create_new = False

        if can_create_new:
            new_spk_id = f"SPK{self._next_spk_idx}"
            self._next_spk_idx += 1
            self.clusters[new_spk_id] = SpeakerCluster(
                speaker_id=new_spk_id,
                centroid=embedding.copy(),
                count=1,
            )
            return new_spk_id
        else:
            # Forced assignment to closest cluster to prevent speaker ID explosion
            self.clusters[best_spk_id].update(embedding)
            return best_spk_id
