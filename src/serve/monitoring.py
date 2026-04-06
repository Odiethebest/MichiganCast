from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class MonitoringConfig:
    log_jsonl_path: str = "artifacts/logs/inference_monitoring.jsonl"
    max_history_points: int = 5000


class InferenceMonitor:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self._lock = threading.Lock()
        self._request_count = 0
        self._sample_count = 0
        self._latency_ms: List[float] = []
        self._prediction_scores: List[float] = []
        self._image_means: List[float] = []
        self._image_stds: List[float] = []
        self._meteo_means: List[float] = []
        self._meteo_stds: List[float] = []

    def _append_with_cap(self, container: List[float], values: List[float]) -> None:
        container.extend(values)
        overflow = len(container) - self.config.max_history_points
        if overflow > 0:
            del container[:overflow]

    def _write_event(self, event: Dict[str, object]) -> None:
        log_path = Path(self.config.log_jsonl_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record(
        self,
        *,
        request_id: str,
        image_array: np.ndarray,
        meteo_array: np.ndarray,
        prediction_scores: np.ndarray,
        latency_ms: float,
        model_path: str,
    ) -> None:
        image_means = [float(np.mean(image_array))]
        image_stds = [float(np.std(image_array))]
        meteo_means = [float(np.mean(meteo_array))]
        meteo_stds = [float(np.std(meteo_array))]
        pred_scores = prediction_scores.astype(float).flatten().tolist()

        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_path": model_path,
            "batch_size": int(image_array.shape[0]),
            "image_shape": list(image_array.shape),
            "meteo_shape": list(meteo_array.shape),
            "latency_ms": float(latency_ms),
            "image_mean": image_means[0],
            "image_std": image_stds[0],
            "meteo_mean": meteo_means[0],
            "meteo_std": meteo_stds[0],
            "prediction_scores": pred_scores,
            "prediction_mean": float(np.mean(prediction_scores)),
            "prediction_min": float(np.min(prediction_scores)),
            "prediction_max": float(np.max(prediction_scores)),
        }

        with self._lock:
            self._request_count += 1
            self._sample_count += int(image_array.shape[0])
            self._append_with_cap(self._latency_ms, [float(latency_ms)])
            self._append_with_cap(self._prediction_scores, pred_scores)
            self._append_with_cap(self._image_means, image_means)
            self._append_with_cap(self._image_stds, image_stds)
            self._append_with_cap(self._meteo_means, meteo_means)
            self._append_with_cap(self._meteo_stds, meteo_stds)
            self._write_event(event)

    def summary(self) -> Dict[str, object]:
        with self._lock:
            lat = np.asarray(self._latency_ms, dtype=float) if self._latency_ms else np.asarray([0.0])
            pred = (
                np.asarray(self._prediction_scores, dtype=float)
                if self._prediction_scores
                else np.asarray([0.0])
            )
            hist_counts, hist_bins = np.histogram(pred, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            payload = {
                "request_count": int(self._request_count),
                "sample_count": int(self._sample_count),
                "latency_ms": {
                    "mean": float(np.mean(lat)),
                    "p50": float(np.percentile(lat, 50)),
                    "p95": float(np.percentile(lat, 95)),
                    "max": float(np.max(lat)),
                },
                "prediction_score": {
                    "mean": float(np.mean(pred)),
                    "std": float(np.std(pred)),
                    "min": float(np.min(pred)),
                    "max": float(np.max(pred)),
                    "histogram": {
                        "bins": [float(x) for x in hist_bins.tolist()],
                        "counts": [int(x) for x in hist_counts.tolist()],
                    },
                },
                "input_distribution": {
                    "image_mean_mean": float(np.mean(self._image_means)) if self._image_means else 0.0,
                    "image_std_mean": float(np.mean(self._image_stds)) if self._image_stds else 0.0,
                    "meteo_mean_mean": float(np.mean(self._meteo_means)) if self._meteo_means else 0.0,
                    "meteo_std_mean": float(np.mean(self._meteo_stds)) if self._meteo_stds else 0.0,
                },
                "log_jsonl_path": self.config.log_jsonl_path,
            }
        return payload
