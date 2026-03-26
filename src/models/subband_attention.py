from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class SubbandAttentionClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        n_subbands: int = 4,
        subband_channels: int = 64,
        classifier_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        if n_mels != 64 or n_subbands != 4:
            raise ValueError("Current implementation supports 64 mel bins and 4 subbands.")

        self.subbands: List[Tuple[int, int]] = [(0, 16), (16, 32), (32, 48), (48, 64)]
        self.subband_cnns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, subband_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(subband_channels),
                    nn.ReLU(),
                    nn.Conv2d(subband_channels, subband_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(subband_channels),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                for _ in range(n_subbands)
            ]
        )

        self.subband_attention = nn.Sequential(
            nn.Linear(subband_channels * n_subbands, 128),
            nn.ReLU(),
            nn.Linear(128, n_subbands),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(subband_channels * n_subbands, classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        sub_feats = []
        for cnn, (s, e) in zip(self.subband_cnns, self.subbands):
            sb = x[:, :, s:e, :]
            feat = cnn(sb).squeeze(-1).squeeze(-1)
            sub_feats.append(feat)

        all_features = torch.cat(sub_feats, dim=1)
        attn = self.subband_attention(all_features)
        logits = self.classifier(all_features).squeeze(-1)

        return {
            "logits": logits,
            "subband_attention_weights": attn,
            "embeddings": all_features,
        }
