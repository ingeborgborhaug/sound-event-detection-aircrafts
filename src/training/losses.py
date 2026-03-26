from typing import Dict, Optional

import torch
import torch.nn as nn


class SEDLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, domain_adaptation_weight: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.domain_adaptation_weight = float(domain_adaptation_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, model_output: Dict[str, torch.Tensor], targets: torch.Tensor, domain_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = model_output["logits"]
        y = targets.float()
        if self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = self.bce(logits, y)
        domain_loss = torch.tensor(0.0, device=logits.device)
        if "domain_loss" in model_output and domain_labels is not None:
            domain_loss = model_output["domain_loss"]

        total_loss = bce_loss + self.domain_adaptation_weight * domain_loss
        return {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "domain_loss": domain_loss,
        }
