import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Phase 12 thesis figure/table generation")
    p.add_argument("--outputs-root", type=str, default="outputs")
    p.add_argument("--out-dir", type=str, default="outputs/thesis_figures")
    return p.parse_args()


def _save_table_with_latex(df: pd.DataFrame, out_csv: Path, out_tex: Path, caption: str = ""):
    df.to_csv(out_csv, index=False)
    latex = df.to_latex(index=False, caption=caption, bold_rows=False)
    out_tex.write_text(latex, encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 11})

    # Minimal robust implementation: aggregate available experiment summaries.
    rows = []
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        summary_csv = exp_dir / "evaluation_report" / "per_condition_metrics.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            df["experiment"] = exp_dir.name
            rows.append(df)

    if rows:
        comp = pd.concat(rows, ignore_index=True)
        _save_table_with_latex(comp, out_dir / "table_main_results.csv", out_dir / "table_main_results.tex", caption="Main results")

        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = comp.pivot_table(index="session", columns="experiment", values="f1")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("F1")
        ax.set_title("Model comparison by weather condition")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_model_comparison.png", dpi=300)
        fig.savefig(out_dir / "fig_model_comparison.pdf")

    print(f"Saved thesis figures/tables to: {out_dir}")


if __name__ == "__main__":
    main()
