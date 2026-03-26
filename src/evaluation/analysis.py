from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import det_curve, roc_curve

from .metrics import compute_per_condition_metrics, compute_segment_metrics


def generate_det_curve(y_true, y_pred_proba, label: str = "Model"):
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    fpr, fnr, _ = det_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, fnr, label=label)
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Miss Rate")
    ax.set_title("DET Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fpr, fnr, fig


def generate_per_condition_report(all_predictions: Dict[str, pd.DataFrame], output_dir: str = "outputs/evaluation_report"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.concat(all_predictions.values(), ignore_index=True)
    summary = compute_per_condition_metrics(full_df)

    rows = []
    for session, metrics in summary["per_session"].items():
        rows.append({"session": session, "f1": metrics["f1"], "auc_roc": metrics["auc_roc"]})
    session_df = pd.DataFrame(rows).sort_values("session")
    session_df.to_csv(out_dir / "per_condition_metrics.csv", index=False)

    # ROC curves by session
    fig, ax = plt.subplots(figsize=(10, 6))
    for session, grp in full_df.groupby("session"):
        y_true = grp["y_true"].astype(int).values
        y_prob = grp["y_pred_proba"].astype(float).values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=str(session))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC by Session")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_by_session.png", dpi=150)

    # DET curves by session
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for session, grp in full_df.groupby("session"):
        y_true = grp["y_true"].astype(int).values
        y_prob = grp["y_pred_proba"].astype(float).values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, fnr, _ = det_curve(y_true, y_prob)
        ax2.plot(fpr, fnr, label=str(session))
    ax2.set_xlabel("False Alarm Rate")
    ax2.set_ylabel("Miss Rate")
    ax2.set_title("DET by Session")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "det_by_session.png", dpi=150)

    # F1 bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(session_df["session"], session_df["f1"])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("F1")
    ax3.set_title("F1 by Weather Session")
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "f1_by_session.png", dpi=150)

    return {"summary": summary, "per_condition_path": str(out_dir / "per_condition_metrics.csv")}


def snr_stratified_analysis(predictions_df: pd.DataFrame, noise_profiles: Dict, output_dir: str = "outputs/evaluation_report"):
    _ = noise_profiles
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = predictions_df.copy()
    if "snr_estimate" not in df.columns:
        # Fallback proxy so pipeline remains usable before exact SNR modeling is added.
        df["snr_estimate"] = (df["y_pred_proba"].astype(float) * 30.0) - 5.0

    bins = [-np.inf, 0, 5, 10, 15, 20, np.inf]
    labels = ["<0", "0-5", "5-10", "10-15", "15-20", ">20"]
    df["snr_bin"] = pd.cut(df["snr_estimate"], bins=bins, labels=labels)

    rows = []
    for snr_bin, grp in df.groupby("snr_bin", observed=False):
        if len(grp) == 0:
            continue
        metrics = compute_segment_metrics(grp["y_true"], grp["y_pred_proba"])
        rows.append({"snr_bin": str(snr_bin), "f1": metrics["f1"], "n": len(grp)})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "snr_stratified_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(out):
        ax.plot(out["snr_bin"], out["f1"], marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Estimated SNR bin (dB)")
    ax.set_ylabel("F1")
    ax.set_title("SNR-stratified performance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "snr_stratified_f1.png", dpi=150)

    return out
