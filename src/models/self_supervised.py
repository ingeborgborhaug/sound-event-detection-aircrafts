from typing import Dict

import torch
import torch.nn as nn


class MaskedSpectrogramModel(nn.Module):
    def __init__(self, encoder: nn.Module, mask_ratio: float = 0.3, n_mels: int = 64, patch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = float(mask_ratio)
        self.n_mels = int(n_mels)
        self.patch_size = int(patch_size)

        # Decoder expects encoder output channels exposed via `out_channels` if available.
        embed_dim = int(getattr(encoder, "out_channels", 128))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _mask(self, x: torch.Tensor):
        b, c, h, w = x.shape
        mask = torch.rand((b, 1, h, w), device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask.expand_as(x_masked)] = 0.0
        return x_masked, mask

    def forward(self, spectrogram: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_masked, mask = self._mask(spectrogram)
        feats = self.encoder(x_masked)
        recon = self.decoder(feats)
        recon = torch.nn.functional.interpolate(recon, size=spectrogram.shape[-2:], mode="bilinear", align_corners=False)

        sq = (recon - spectrogram) ** 2
        loss = sq[mask.expand_as(sq)].mean() if mask.any() else sq.mean()
        return {"reconstruction_loss": loss, "encoder_features": feats}
