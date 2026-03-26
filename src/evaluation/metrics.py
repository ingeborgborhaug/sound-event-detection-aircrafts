from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def compute_segment_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_pred_proba).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / max(tn + fp, 1))

    auc_roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "specificity": spec,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
        "optimal_threshold": _optimal_threshold(y_true, y_prob),
    }


def compute_metrics_at_optimal_threshold(y_true, y_pred_proba) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_pred_proba).astype(float)
    thr = _optimal_threshold(y_true, y_prob)
    out = compute_segment_metrics(y_true, y_prob, threshold=thr)
    out["optimal_threshold"] = thr
    return out


def compute_per_condition_metrics(predictions_df: pd.DataFrame):
    req = {"session", "y_true", "y_pred_proba"}
    missing = req - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    by_session = {}
    for session, grp in predictions_df.groupby("session"):
        by_session[str(session)] = compute_metrics_at_optimal_threshold(grp["y_true"], grp["y_pred_proba"])

    overall = compute_metrics_at_optimal_threshold(predictions_df["y_true"], predictions_df["y_pred_proba"])
    return {"per_session": by_session, "overall": overall}
