from __future__ import annotations

from pathlib import Path

import tensorflow as tf

import settings
from keras_yamnet.params import PATCH_BANDS, PATCH_FRAMES
from keras_yamnet.yamnet import YAMNet
from src.models.classifier_heads import CustomClassificationHead, TemporalAttentionHead


def build_yamnet_classifier(
    yamnet_weights: str | Path = "keras_yamnet/yamnet.h5",
    freeze_backbone: bool = True,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> tf.keras.Model:
    """Model for Option B: cached YAMNet mel patches -> YAMNet encoder -> temporal head."""

    yamnet_weights = str(yamnet_weights)
    try:
        backbone = YAMNet(include_top=False, pooling="avg", weights=yamnet_weights)
    except ValueError:
        full_model = YAMNet(include_top=True, weights=yamnet_weights)
        # The embedding is the global average pooling output before classifier dense+activation.
        backbone = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.layers[-3].output,
            name="yamnet_backbone_embedder",
        )
    backbone.trainable = not freeze_backbone

    x_in = tf.keras.Input(shape=(PATCH_FRAMES, PATCH_BANDS), name="patches")

    embeds = backbone(x_in)

    out = CustomClassificationHead()(embeds)
    model = tf.keras.Model(inputs=x_in, outputs=out, name="yamnet_classifier")
    return model


YAMNetClassifier = build_yamnet_classifier
