from itertools import combinations
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(y_pred_a).astype(int)
    b = np.asarray(y_pred_b).astype(int)

    a_ok = a == y_true
    b_ok = b == y_true

    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())

    if (n01 + n10) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return {"chi2": float(chi2_stat), "p_value": float(p_value), "significant": bool(p_value < 0.05)}


def bootstrap_ci(
    y_true,
    y_pred_proba,
    metric_fn: Callable,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_pred_proba)
    rng = np.random.default_rng(seed)

    vals = []
    n = len(y_true)
    for _ in range(int(n_bootstraps)):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_prob[idx])))

    vals = np.asarray(vals)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(vals, alpha))
    upper = float(np.quantile(vals, 1.0 - alpha))
    return lower, upper, float(vals.mean()), float(vals.std())


def compare_conditions_significance(per_condition_predictions: Dict[str, pd.DataFrame]):
    names = sorted(per_condition_predictions.keys())
    pvals = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)

    for a, b in combinations(names, 2):
        dfa = per_condition_predictions[a]
        dfb = per_condition_predictions[b]

        ya = (dfa["y_pred_proba"].astype(float).values >= 0.5).astype(int)
        yb = (dfb["y_pred_proba"].astype(float).values >= 0.5).astype(int)
        yt = dfa["y_true"].astype(int).values

        n = min(len(yt), len(ya), len(yb))
        res = mcnemar_test(yt[:n], ya[:n], yb[:n])
        pvals.loc[a, b] = res["p_value"]
        pvals.loc[b, a] = res["p_value"]

    # Bonferroni correction
    m = max(len(list(combinations(names, 2))), 1)
    corrected = (pvals * m).clip(upper=1.0)
    return corrected
