from typing import Iterable

import coremltools as ct
from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
from numpy.random import rand

from . import keras, red


def random_fit(
    model: keras.Model, in_shape: Iterable[int], out_shape: Iterable[int] = (1,)
):
    x = rand(1, *in_shape)
    y = rand(1, *out_shape)
    model.fit(x, y, epochs=1)


def convert(model: keras.Model) -> MLModel:
    return ct.convert(model)  # type: ignore


def nn_builder(mlmodel: MLModel) -> NeuralNetworkBuilder:
    spec = mlmodel.get_spec()
    return NeuralNetworkBuilder(spec=spec)


def try_make_layers_updatable(builder: NeuralNetworkBuilder, limit_last: int = 0xFFFF):
    assert builder.nn_spec is not None
    updatable_layer_names: list[str] = []
    updatable_layers: list[dict[str, str | list[int]]] = []
    for layer in builder.nn_spec.layers:
        name = layer.name
        kind = layer.WhichOneof("layer")
        if kind == "convolution":
            layer = layer.convolution
        elif kind == "innerProduct":
            layer = layer.innerProduct
        else:
            continue
        updatable_layer_names.append(name)
        updatable_layers.append({"name": name, "type": "weights"})
        info = f"{name}: Weights: {len(layer.weights.floatValue) * 4} bytes"
        if layer.hasBias:
            updatable_layers.append({"name": name, "type": "bias"})
            info += f", Bias: {len(layer.bias.floatValue) * 4} bytes"
        print(f"{info}.")
    made_updatable = updatable_layer_names[-limit_last:]
    builder.make_updatable(made_updatable)
    print(f"Made {made_updatable} updatable.")
    print(f"All updatable layers:\n\t{red(updatable_layers)}")
    return updatable_layers


def save_builder(builder: NeuralNetworkBuilder, directory: str):
    mlmodel = MLModel(builder.spec)
    mlmodel.save(directory)
