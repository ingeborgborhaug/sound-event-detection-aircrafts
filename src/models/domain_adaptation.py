from typing import Dict

import torch
import torch.nn as nn


class DomainAdaptationModel(nn.Module):
    def __init__(self, base_model: nn.Module, adaptation_method: str = "coral", adaptation_weight: float = 1.0):
        super().__init__()
        if adaptation_method not in {"coral", "mmd"}:
            raise ValueError("adaptation_method must be 'coral' or 'mmd'")
        self.base_model = base_model
        self.adaptation_method = adaptation_method
        self.adaptation_weight = float(adaptation_weight)

    @staticmethod
    def coral_loss(source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        source = source_embeddings - source_embeddings.mean(dim=0, keepdim=True)
        target = target_embeddings - target_embeddings.mean(dim=0, keepdim=True)
        ns, nt = source.shape[0], target.shape[0]
        cs = (source.T @ source) / max(ns - 1, 1)
        ct = (target.T @ target) / max(nt - 1, 1)
        d = source.shape[1]
        return ((cs - ct) ** 2).sum() / (4.0 * d * d)

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        dist2 = torch.cdist(x, y, p=2.0) ** 2
        return torch.exp(-gamma * dist2)

    def mmd_loss(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        z = torch.cat([source_embeddings, target_embeddings], dim=0)
        with torch.no_grad():
            d2 = torch.cdist(z, z, p=2.0) ** 2
            med = torch.median(d2[d2 > 0]) if torch.any(d2 > 0) else torch.tensor(1.0, device=z.device)
            base_gamma = 1.0 / (med + 1e-6)
        gammas = [base_gamma / 2.0, base_gamma, base_gamma * 2.0]
        mmd = torch.tensor(0.0, device=z.device)
        for g in gammas:
            k_ss = self._rbf_kernel(source_embeddings, source_embeddings, float(g)).mean()
            k_tt = self._rbf_kernel(target_embeddings, target_embeddings, float(g)).mean()
            k_st = self._rbf_kernel(source_embeddings, target_embeddings, float(g)).mean()
            mmd = mmd + (k_ss + k_tt - 2.0 * k_st)
        return mmd / len(gammas)

    def forward(self, x: torch.Tensor, domain_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        out = self.base_model(x)

        if domain_labels is None or "embeddings" not in out:
            out["domain_loss"] = torch.tensor(0.0, device=x.device)
            return out

        source_mask = domain_labels == 0
        target_mask = domain_labels == 1
        if source_mask.sum() > 1 and target_mask.sum() > 1:
            src = out["embeddings"][source_mask]
            tgt = out["embeddings"][target_mask]
            if self.adaptation_method == "coral":
                d_loss = self.coral_loss(src, tgt)
            else:
                d_loss = self.mmd_loss(src, tgt)
            out["domain_loss"] = d_loss * self.adaptation_weight
        else:
            out["domain_loss"] = torch.tensor(0.0, device=x.device)

        return out
