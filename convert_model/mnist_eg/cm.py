from coremltools.models import datatypes
from coremltools.models.neural_network import AdamParams, NeuralNetworkBuilder

from ..coreml import (
    convert,
    nn_builder,
    random_fit,
    save_builder,
    try_make_layers_updatable,
)
from . import in_shape, mnist_model, n_classes

COREML_FILE = "mnist.mlmodel"


def config_builder(builder: NeuralNetworkBuilder):
    builder.set_mean_squared_error_loss(
        "lossLayer",
        input_feature=("Identity", datatypes.Array(n_classes)),
    )
    builder.set_adam_optimizer(AdamParams())
    max_epochs = 10
    builder.set_epochs(max_epochs, range(1, max_epochs + 1))


def main():
    model = mnist_model()
    random_fit(model, in_shape, (n_classes,))
    mlmodel = convert(model)
    builder = nn_builder(mlmodel)
    config_builder(builder)
    try_make_layers_updatable(builder, 2)
    builder.inspect_layers()
    save_builder(builder, COREML_FILE)


if __name__ == "__main__":
    main()
