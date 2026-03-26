import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_segment_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Phase 9 multi-location fusion")
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs/fusion")
    return p.parse_args()


def _fuse_probs(arr: np.ndarray, method: str, threshold: float = 0.5) -> float:
    if method == "mean_prob":
        return float(arr.mean())
    if method == "max_prob":
        return float(arr.max())
    if method == "majority_vote":
        return float((arr >= threshold).mean() >= 0.5)
    if method == "product":
        return float(np.prod(np.clip(arr, 1e-6, 1.0)) ** (1.0 / len(arr)))
    if method == "minimum":
        return float(arr.min())
    raise ValueError(f"Unknown fusion method: {method}")


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)
    req = {"session", "time_start", "time_end", "location", "pred_prob", "true_label"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    methods = ["mean_prob", "max_prob", "majority_vote", "product", "minimum"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fused_rows = []
    for method in methods:
        probs = []
        y = []
        for _, grp in df.groupby(["session", "time_start", "time_end"]):
            arr = grp["pred_prob"].astype(float).values
            p = _fuse_probs(arr, method=method)
            probs.append(p)
            y.append(int(grp["true_label"].iloc[0]))
            fused_rows.append(
                {
                    "method": method,
                    "session": grp["session"].iloc[0],
                    "time_start": grp["time_start"].iloc[0],
                    "time_end": grp["time_end"].iloc[0],
                    "pred_prob": p,
                    "pred_label": int(p >= 0.5),
                    "true_label": int(grp["true_label"].iloc[0]),
                }
            )

        m = compute_segment_metrics(y, probs, threshold=0.5)
        rows.append({"method": method, "f1": m["f1"], "auc_roc": m["auc_roc"], "precision": m["precision"], "recall": m["recall"]})

    pd.DataFrame(rows).to_csv(out_dir / "fusion_metrics.csv", index=False)
    pd.DataFrame(fused_rows).to_csv(out_dir / "fused_predictions.csv", index=False)
    print(f"Saved fusion outputs to: {out_dir}")


if __name__ == "__main__":
    main()
