from .. import keras
from ..tflite import BaseTFLiteModel, tflite_model_class


@tflite_model_class
class FedMCRNNModel(BaseTFLiteModel):
    X_SHAPE = [7, 8]
    Y_SHAPE = [1]

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        """Written and tuned by Aicha Slaitane in Aug 2023."""
        model = keras.Sequential()
        # For the first LSTM layer, specify the input_shape
        model.add(
            keras.layers.LSTM(
                # Tune number of units separately.
                units=384,
                input_shape=self.X_SHAPE,
                return_sequences=True,
            )
        )
        model.add(keras.layers.LeakyReLU(0.523629795960645))
        model.add(keras.layers.Dropout(0.372150795833))

        # For subsequent LSTM layers, no need to specify input_shape
        model.add(
            keras.layers.LSTM(
                units=64,
                return_sequences=True,
            )
        )
        model.add(keras.layers.LeakyReLU(0.523629795960645))
        model.add(keras.layers.Dropout(0.372150795833))

        model.add(
            keras.layers.LSTM(
                units=480,
                return_sequences=True,
            )
        )
        model.add(keras.layers.LeakyReLU(0.523629795960645))
        model.add(keras.layers.Dropout(0.372150795833))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00668472266354),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )
        return model
