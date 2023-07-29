from typing import Callable

import tensorflow as tf

from .. import *


def repeat(fn: Callable[[], list], times: int):
    return sum((fn() for _ in range(times)), [])


def repeat3(fn: Callable[[], list]):
    return repeat(fn, 3)


def conv_layers():
    return [
        tf.keras.layers.Conv1D(7, kernel_size=3),
        tf.keras.layers.BatchNormalization(),
    ]


def recurrent_layers():
    return [tf.keras.layers.LSTM(7, dropout=0.2, return_sequences=True)]


@tflite_model_class
class FedMCRNNModel(BaseTFLiteModel):
    X_SHAPE = [7, 8]
    Y_SHAPE = [1]

    def __init__(self):
        self.model = tf.keras.models.Sequential(
            [tf.keras.Input(shape=tuple(self.X_SHAPE))]
            + repeat3(conv_layers)
            + [
                tf.keras.layers.Dense(56, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.RepeatVector(7),
            ]
            + repeat3(recurrent_layers)
            + [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1),
            ]
        )
        self.model.compile(loss=tf.keras.losses.MeanSquaredError())
