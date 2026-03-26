from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class EmbeddingLogisticRegression(nn.Module):
    def __init__(self, embedding_dim: int = 1024, aggregation: str = "mean"):
        super().__init__()
        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.aggregation = aggregation
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, n_patches, emb), got {tuple(x.shape)}")
        pooled = x.mean(dim=1) if self.aggregation == "mean" else x.max(dim=1).values
        logits = self.fc(pooled).squeeze(-1)
        return {"logits": logits, "embeddings": pooled}


class TemporalClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        n_layers: int = 2,
        temporal_model: str = "gru",
        use_attention: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_model = temporal_model
        self.use_attention = bool(use_attention)

        if temporal_model == "gru":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=True,
            )
            out_dim = hidden_dim * 2
        elif temporal_model == "attention":
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            out_dim = embedding_dim
        else:
            raise ValueError("temporal_model must be 'gru' or 'attention'")

        self.pool = AttentionPooling(out_dim) if self.use_attention else None
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, n_patches, emb), got {tuple(x.shape)}")

        if self.temporal_model == "gru":
            seq, _ = self.rnn(x)
        else:
            seq = self.encoder(x)

        if self.pool is not None:
            pooled = self.pool(seq)
        else:
            pooled = seq[:, -1, :]

        logits = self.classifier(pooled).squeeze(-1)
        return {"logits": logits, "embeddings": pooled}


class SpectrogramPatchClassifier(nn.Module):
    """Simple CNN head for spectrogram patches shaped (B, 1, n_mels, n_frames)."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected (batch, 1, n_mels, n_frames), got {tuple(x.shape)}")
        feats = self.backbone(x).flatten(1)
        logits = self.classifier[:-1](feats)
        logits = self.classifier[-1](logits).squeeze(-1)
        return {"logits": logits, "embeddings": feats}
