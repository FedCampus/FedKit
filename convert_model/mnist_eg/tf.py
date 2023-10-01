from ..tflite import (
    SAVED_MODEL_DIR,
    BaseTFLiteModel,
    convert_saved_model,
    save_model,
    save_tflite_model,
    tflite_model_class,
)
from . import in_shape, mnist_model

TFLITE_FILE = "mnist.tflite"


@tflite_model_class
class MnistTFModel(BaseTFLiteModel):
    X_SHAPE = list(in_shape)
    Y_SHAPE = [10]

    def __init__(self):
        self.model = mnist_model()


def tflite():
    model = MnistTFModel()
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)
    print(f"Successfully converted to TFLite model at {TFLITE_FILE}.")


if __name__ == "__main__":
    tflite()
