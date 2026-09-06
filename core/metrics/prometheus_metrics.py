import time
import threading
from typing import Dict, List, Any, Optional, Tuple


class PrometheusMetricsCollector:
    """
    Lightweight, thread-safe Prometheus metrics collector and text exporter.
    Tracks all key real-time ASR, VAD, Speaker, Queue, and Guard metrics.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PrometheusMetricsCollector, cls).__new__(cls)
                cls._instance._init_metrics()
            return cls._instance

    def _init_metrics(self):
        self._gauges: Dict[str, float] = {
            "asr_online_sessions": 0.0,
            "asr_active_speech_sessions": 0.0,
            "asr_qwen_queue_size": 0.0,
            "asr_streaming_engine_reloading": 0.0,
            "asr_vad_engine_reloading": 0.0,
            "asr_admission_saturated": 0.0,
            "asr_admission_streaming_wait_ms": 0.0,
            "asr_admission_vad_wait_ms": 0.0,
            "asr_admission_final_fallback_rate": 0.0,
        }
        self._counters: Dict[str, float] = {
            "asr_total_sessions": 0.0,
            "asr_websocket_errors": 0.0,
            "asr_qwen_timeout_total": 0.0,
            "asr_qwen_fallback_total": 0.0,
            "asr_sentences_finalized": 0.0,
            "asr_sentences_needs_review": 0.0,
            "asr_sessions_evicted_idle": 0.0,
            "asr_streaming_engine_errors": 0.0,
            "asr_streaming_engine_reloads": 0.0,
            "asr_vad_engine_errors": 0.0,
            "asr_vad_engine_reloads": 0.0,
            "asr_malformed_frames_total": 0.0,
            "asr_chunk_processing_errors_total": 0.0,
            "asr_sessions_rejected_overload_total": 0.0,
            "asr_silence_dropped_chunks_total": 0.0,
            "asr_partial_suppressed_total": 0.0,
            "asr_degenerate_ngram_detected_total": 0.0,
        }
        # name -> {sorted label pairs -> value}
        self._labeled_counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {
            "asr_watchdog_force_total": {
                (("result", "force"),): 0.0,
                (("result", "ratelimit"),): 0.0,
            },
        }
        self._latencies: Dict[str, List[float]] = {
            "asr_paraformer_latency_ms": [],
            "asr_vad_latency_ms": [],
            "asr_vad_streaming_latency_ms": [],
            "asr_qwen_queue_wait_ms": [],
            "asr_qwen_inference_ms": [],
            "asr_speaker_embedding_ms": [],
            "asr_revision_distance": [],
            "asr_qwen_batch_size": [],
            "asr_event_loop_lag_ms": [],
            "asr_vad_lock_wait_ms": [],
            "asr_partial_text_len": [],
        }
        self._metric_lock = threading.Lock()

    def set_gauge(self, name: str, value: float):
        with self._metric_lock:
            self._gauges[name] = float(value)

    def inc_gauge(self, name: str, delta: float = 1.0):
        with self._metric_lock:
            self._gauges[name] = self._gauges.get(name, 0.0) + delta

    def dec_gauge(self, name: str, delta: float = 1.0):
        with self._metric_lock:
            self._gauges[name] = max(0.0, self._gauges.get(name, 0.0) - delta)

    def inc_counter(self, name: str, delta: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self._metric_lock:
            if labels:
                key = tuple(sorted(labels.items()))
                bucket = self._labeled_counters.setdefault(name, {})
                bucket[key] = bucket.get(key, 0.0) + delta
            else:
                self._counters[name] = self._counters.get(name, 0.0) + delta

    def observe(self, name: str, value: float):
        with self._metric_lock:
            if name not in self._latencies:
                self._latencies[name] = []
            buf = self._latencies[name]
            buf.append(float(value))
            if len(buf) > 1000:  # Keep recent 1000 observations
                self._latencies[name] = buf[-1000:]

    def export_text(self) -> str:
        """
        Export metrics in Prometheus standard plain-text format.
        """
        lines: List[str] = []
        with self._metric_lock:
            # Gauges
            for name, val in self._gauges.items():
                lines.append(f"# HELP {name} Current gauge value")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {val}")

            # Counters
            for name, val in self._counters.items():
                lines.append(f"# HELP {name} Total counter value")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")

            for name, series in self._labeled_counters.items():
                lines.append(f"# HELP {name} Total counter value")
                lines.append(f"# TYPE {name} counter")
                for key, val in series.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in key)
                    lines.append(f"{name}{{{label_str}}} {val}")

            # Summaries / Histograms
            for name, values in self._latencies.items():
                lines.append(f"# HELP {name} Observed values")
                lines.append(f"# TYPE {name} summary")
                if values:
                    count = len(values)
                    total = sum(values)
                    sorted_v = sorted(values)
                    p50 = sorted_v[int(count * 0.50)]
                    p90 = sorted_v[min(count - 1, int(count * 0.90))]
                    p95 = sorted_v[min(count - 1, int(count * 0.95))]
                    p99 = sorted_v[min(count - 1, int(count * 0.99))]

                    lines.append(f'{name}{{quantile="0.50"}} {p50:.2f}')
                    lines.append(f'{name}{{quantile="0.90"}} {p90:.2f}')
                    lines.append(f'{name}{{quantile="0.95"}} {p95:.2f}')
                    lines.append(f'{name}{{quantile="0.99"}} {p99:.2f}')
                    lines.append(f"{name}_sum {total:.2f}")
                    lines.append(f"{name}_count {count}")
                else:
                    lines.append(f"{name}_sum 0.0")
                    lines.append(f"{name}_count 0")

        return "\n".join(lines) + "\n"


metrics = PrometheusMetricsCollector()
