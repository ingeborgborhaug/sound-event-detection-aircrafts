from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def predictions_to_event_list(times: Sequence[Tuple[float, float]], labels: Sequence[int], min_event_duration_s: float = 0.5):
    events: List[Tuple[float, float]] = []
    cur_start = None
    cur_end = None

    for (t0, t1), y in zip(times, labels):
        if int(y) == 1:
            if cur_start is None:
                cur_start, cur_end = float(t0), float(t1)
            else:
                cur_end = float(t1)
        else:
            if cur_start is not None and (cur_end - cur_start) >= min_event_duration_s:
                events.append((cur_start, cur_end))
            cur_start, cur_end = None, None

    if cur_start is not None and (cur_end - cur_start) >= min_event_duration_s:
        events.append((cur_start, cur_end))

    return events


def _event_match_stats(ref_events, est_events, collar_s=0.2):
    matched_ref = set()
    matched_est = set()

    for i, (r0, r1) in enumerate(ref_events):
        for j, (e0, e1) in enumerate(est_events):
            if j in matched_est:
                continue
            if abs(r0 - e0) <= collar_s and abs(r1 - e1) <= collar_s:
                matched_ref.add(i)
                matched_est.add(j)
                break

    tp = len(matched_ref)
    fp = len(est_events) - tp
    fn = len(ref_events) - tp
    return tp, fp, fn


def compute_event_metrics(predictions_df: pd.DataFrame, collar_s: float = 0.2, t_collar_s: float = 0.2):
    _ = t_collar_s
    req = {"start_s", "end_s", "y_true", "y_pred_proba"}
    missing = req - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = predictions_df.sort_values(["start_s", "end_s"]).reset_index(drop=True)
    times = list(zip(df["start_s"].astype(float).tolist(), df["end_s"].astype(float).tolist()))

    y_true = df["y_true"].astype(int).tolist()
    y_pred = (df["y_pred_proba"].astype(float).values >= 0.5).astype(int).tolist()

    ref_events = predictions_to_event_list(times, y_true)
    est_events = predictions_to_event_list(times, y_pred)

    tp, fp, fn = _event_match_stats(ref_events, est_events, collar_s=collar_s)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # Approximate segment metrics from binary frame predictions.
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    seg_tp = int(((yt == 1) & (yp == 1)).sum())
    seg_fp = int(((yt == 0) & (yp == 1)).sum())
    seg_fn = int(((yt == 1) & (yp == 0)).sum())
    seg_prec = seg_tp / max(seg_tp + seg_fp, 1)
    seg_rec = seg_tp / max(seg_tp + seg_fn, 1)
    seg_f1 = 2 * seg_prec * seg_rec / max(seg_prec + seg_rec, 1e-12)
    seg_er = (seg_fp + seg_fn) / max(int((yt == 1).sum()), 1)

    return {
        "segment_based_f1": float(seg_f1),
        "segment_based_er": float(seg_er),
        "event_based_f1": float(f1),
        "event_based_precision": float(prec),
        "event_based_recall": float(rec),
        "deletion_rate": float(fn / max(len(ref_events), 1)),
        "insertion_rate": float(fp / max(len(ref_events), 1)),
    }
