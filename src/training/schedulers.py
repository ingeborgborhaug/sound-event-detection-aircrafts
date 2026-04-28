from __future__ import annotations

import tensorflow as tf


def cosine_decay(initial_lr: float, epochs: int) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    return tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr, decay_steps=max(1, epochs))
