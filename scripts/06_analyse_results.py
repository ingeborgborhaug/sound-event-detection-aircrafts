import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compare experiments")
    p.add_argument("--outputs-root", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.outputs_root)
    exp_dirs = [d for d in root.iterdir() if d.is_dir()]

    rows = []
    for exp in exp_dirs:
        metric_file = exp / "evaluation_report" / "per_condition_metrics.csv"
        if not metric_file.exists():
            continue
        df = pd.read_csv(metric_file)
        df["experiment"] = exp.name
        rows.append(df)

    if not rows:
        raise RuntimeError("No experiment evaluation summaries found.")

    comp = pd.concat(rows, ignore_index=True)
    out_dir = root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = comp.pivot_table(index="session", columns="experiment", values="f1")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("F1")
    ax.set_title("F1 by session across experiments")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_comparison.png", dpi=150)

    print(f"Saved comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()
