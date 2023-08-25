import coremltools as ct
from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.utils import save_spec
from tensorflow import keras


def convert(model: keras.Model) -> MLModel:
    return ct.convert(model)  # type: ignore


def nn_builder(mlmodel: MLModel) -> NeuralNetworkBuilder:
    spec = mlmodel.get_spec()
    return NeuralNetworkBuilder(spec=spec)


def try_make_layers_updatable(builder: NeuralNetworkBuilder):
    assert builder.nn_spec is not None
    for layer in builder.nn_spec.layers:
        name = layer.name
        try:
            builder.make_updatable([name])
            print(f"made {name} updatable")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            print(f"could not make {name} updatable")


def save_builder(builder: NeuralNetworkBuilder, directory: str):
    save_spec(builder.spec, directory)
