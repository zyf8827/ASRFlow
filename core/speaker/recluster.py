from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger


def global_recluster_speakers(
    sentence_records: List[Dict[str, Any]],
    expected_speakers: Optional[int] = None,
    distance_threshold: float = 0.35,  # 1 - similarity_threshold (e.g. 1 - 0.65)
) -> Dict[int, str]:
    """
    Perform global offline speaker re-clustering at session STOP.
    Maps sentence_id -> refined global speaker ID ("SPK1", "SPK2", ...).
    """
    # Extract valid embeddings
    items = []
    for rec in sentence_records:
        emb = rec.get("speaker_embedding")
        s_id = rec.get("sentence_id")
        if emb is not None and s_id is not None:
            items.append((s_id, emb))

    if not items:
        return {}

    # If only 1 sentence
    if len(items) == 1:
        return {items[0][0]: "SPK1"}

    embeddings = np.array([it[1] for it in items], dtype=np.float32)
    # Ensure L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    embeddings = embeddings / norms

    n_samples = len(items)
    n_clusters = None
    if expected_speakers and expected_speakers > 0:
        n_clusters = min(expected_speakers, n_samples)

    labels = None
    try:
        from sklearn.cluster import AgglomerativeClustering

        if n_clusters is not None:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
        labels = clusterer.fit_predict(embeddings)
    except Exception as e:
        # Fallback simple greedy clustering if sklearn is not installed
        logger.debug(f"[Recluster] Using fallback greedy clustering: {e}")
        clusters: List[np.ndarray] = []
        labels = []
        for emb in embeddings:
            if not clusters:
                clusters.append(emb.copy())
                labels.append(0)
                continue
            sims = [float(np.dot(emb, c)) for c in clusters]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= (1.0 - distance_threshold) or (
                n_clusters is not None and len(clusters) >= n_clusters
            ):
                labels.append(best_idx)
            else:
                clusters.append(emb.copy())
                labels.append(len(clusters) - 1)

    # Renumber labels by order of appearance: 0 -> SPK1, 1 -> SPK2, ...
    label_map = {}
    next_id = 1
    result_map: Dict[int, str] = {}

    for (s_id, _), raw_label in zip(items, labels):
        if raw_label not in label_map:
            label_map[raw_label] = f"SPK{next_id}"
            next_id += 1
        result_map[s_id] = label_map[raw_label]

    return result_map
