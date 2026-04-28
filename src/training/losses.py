from __future__ import annotations

import tensorflow as tf


def bce_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.BinaryCrossentropy()
