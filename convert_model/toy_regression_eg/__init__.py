from .. import tf
from ..tflite import BaseTFLiteModel, tflite_model_class


@tflite_model_class
class ToyRegressionModel(BaseTFLiteModel):
    X_SHAPE = [2]
    Y_SHAPE = [1]

    def __init__(self, lr=0.000000001):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=1,
                    input_shape=[2],  # type:ignore
                    name="regression",
                )
            ]
        )

        opt = tf.keras.optimizers.SGD(learning_rate=lr)

        self.model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())
