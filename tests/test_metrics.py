import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics_at_optimal_threshold, compute_segment_metrics


def test_compute_segment_metrics_smoke():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.4])
    out = compute_segment_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= out["f1"] <= 1.0
    assert out["confusion_matrix"].shape == (2, 2)


def test_metrics_at_optimal_threshold_smoke():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.3, 0.7, 0.8, 0.2, 0.55, 0.45])
    out = compute_metrics_at_optimal_threshold(y_true, y_prob)
    assert 0.0 <= out["optimal_threshold"] <= 1.0
    assert 0.0 <= out["f1"] <= 1.0
