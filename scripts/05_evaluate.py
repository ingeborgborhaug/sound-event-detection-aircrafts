from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CV metrics")
    parser.add_argument("--cv-results", required=True, help="Path to cv_results.json")
    args = parser.parse_args()

    results = json.loads(Path(args.cv_results).read_text(encoding="utf-8"))
    keys = sorted({k for r in results for k in r.get("metrics", {}).keys()})

    print("Cross-validation summary")
    for key in keys:
        vals = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        vals = np.array(vals, dtype=float)
        print(f"{key}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")


if __name__ == "__main__":
    main()
