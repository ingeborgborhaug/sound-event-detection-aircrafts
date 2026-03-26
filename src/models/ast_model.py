from typing import Dict

import torch
import torch.nn as nn


class ASTClassifier(nn.Module):
    """
    Lightweight AST-style wrapper using timm ViT backbones.
    NOTE: for lower complexity, consider PANNs or wav2vec2-style audio backbones.
    """

    def __init__(self, pretrained: bool = True, n_classes: int = 1, freeze_backbone: bool = False):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("ASTClassifier requires timm. Install with `pip install timm`.") from exc

        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        embed_dim = getattr(self.backbone, "num_features", 192)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B,1,M,T) -> resize for ViT image input
        x = self.input_adapter(x)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        emb = self.backbone(x)
        logits = self.classifier(emb).squeeze(-1)
        return {"logits": logits, "embeddings": emb}
