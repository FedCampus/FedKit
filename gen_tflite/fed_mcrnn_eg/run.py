from os import path

from .. import *
from . import FedMCRNNModel

DIR = path.dirname(__file__)


TFLITE_FILE = f"fed_mcnrr4.tflite"


def main():
    model = FedMCRNNModel()
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)


main() if __name__ == "__main__" else None
