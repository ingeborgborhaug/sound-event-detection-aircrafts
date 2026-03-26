import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .classifier_heads import EmbeddingLogisticRegression, TemporalClassifier
from .yamnet_embedder import YAMNetEmbedder


class YAMNetFineTune(nn.Module):
    """
    Practical wrapper: frozen TF YAMNet + trainable PyTorch head.
    True end-to-end YAMNet fine-tuning in PyTorch requires reimplementation.
    """

    def __init__(self, freeze_yamnet: bool = True, head_type: str = "temporal", head_config=None):
        super().__init__()
        head_config = head_config or {}
        self.yamnet = YAMNetEmbedder()
        self.freeze_yamnet = bool(freeze_yamnet)

        if not self.freeze_yamnet:
            logging.warning("Requested unfreezing YAMNet, but TF Hub YAMNet is used as frozen feature extractor.")
            self.freeze_yamnet = True

        if head_type == "logistic":
            self.head = EmbeddingLogisticRegression(**head_config)
        elif head_type == "temporal":
            self.head = TemporalClassifier(**head_config)
        else:
            raise ValueError("head_type must be 'logistic' or 'temporal'")

    def forward(self, waveforms_or_embeddings: torch.Tensor, input_type: str = "embeddings") -> Dict[str, torch.Tensor]:
        if input_type == "embeddings":
            return self.head(waveforms_or_embeddings)

        if input_type != "waveform":
            raise ValueError("input_type must be 'embeddings' or 'waveform'")

        with torch.no_grad():
            embs = []
            for wav in waveforms_or_embeddings:
                emb = self.yamnet.extract_embeddings(np.asarray(wav.detach().cpu(), dtype=np.float32))
                embs.append(torch.from_numpy(emb))

            max_len = max(e.shape[0] for e in embs)
            emb_dim = embs[0].shape[1]
            batch = torch.zeros((len(embs), max_len, emb_dim), dtype=torch.float32)
            for i, e in enumerate(embs):
                batch[i, : e.shape[0], :] = e

            batch = batch.to(waveforms_or_embeddings.device)
        return self.head(batch)
