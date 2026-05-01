from __future__ import annotations

import tensorflow as tf


class TemporalAttentionHead(tf.keras.layers.Layer):
    """GRU + attention temporal classification head for patch embeddings."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(hidden_dim, return_sequences=True, dropout=dropout)
        )
        self.attn_score = tf.keras.layers.Dense(1)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.bigru(x, training=training)
        scores = self.attn_score(h)  # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)
        pooled = tf.reduce_sum(weights * h, axis=1)
        return self.classifier(pooled)


class CustomClassificationHead(tf.keras.layers.Layer):
    """Custom classification head with dropout and dense layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)
