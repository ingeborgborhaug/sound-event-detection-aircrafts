from typing import Dict

import pandas as pd
import torch
import torch.nn as nn


class MultiLocationFusion(nn.Module):
    def __init__(self, method: str = "mean", n_locations: int = 3, learned_weights: bool = False):
        super().__init__()
        self.method = "learned" if learned_weights else method
        self.n_locations = int(n_locations)
        if self.method == "learned":
            self.weights = nn.Parameter(torch.ones(self.n_locations) / self.n_locations)

    def forward(self, location_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = torch.stack(list(location_logits.values()), dim=1)
        probs = torch.sigmoid(logits)

        if self.method == "mean":
            p = probs.mean(dim=1)
        elif self.method == "max":
            p = probs.max(dim=1).values
        elif self.method == "majority_vote":
            votes = (probs >= 0.5).float()
            p = (votes.sum(dim=1) >= (votes.shape[1] / 2.0)).float()
        elif self.method == "learned":
            w = torch.softmax(self.weights, dim=0)
            p = (probs * w.unsqueeze(0)).sum(dim=1)
        else:
            raise ValueError("Unknown fusion method")

        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        return torch.logit(p)

    def fuse_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        req = {"session", "time_start", "time_end", "location", "pred_prob", "true_label"}
        missing = req - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        rows = []
        for (session, t0, t1), grp in predictions_df.groupby(["session", "time_start", "time_end"]):
            probs = grp["pred_prob"].astype(float).values
            if self.method == "mean":
                p = float(probs.mean())
            elif self.method == "max":
                p = float(probs.max())
            elif self.method == "majority_vote":
                p = float((probs >= 0.5).mean() >= 0.5)
            elif self.method == "learned":
                w = torch.softmax(self.weights.detach(), dim=0).cpu().numpy()
                use = w[: len(probs)]
                use = use / use.sum()
                p = float((probs * use).sum())
            else:
                raise ValueError("Unknown fusion method")

            rows.append(
                {
                    "session": session,
                    "time_start": t0,
                    "time_end": t1,
                    "pred_prob": p,
                    "pred_label": int(p >= 0.5),
                    "true_label": int(grp["true_label"].iloc[0]),
                }
            )

        return pd.DataFrame(rows)
