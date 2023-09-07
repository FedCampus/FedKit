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
    updatable_layers: list[dict[str, str | list[int]]] = []
    for layer in builder.nn_spec.layers:
        name = layer.name
        try:
            builder.make_updatable([name])
            print(f"made {name} updatable")
            updatable_layers.append({"name": name, "type": "weights"})
            kind = layer.WhichOneof("layer")
            if kind == "convolution":
                if layer.convolution.hasBias:
                    updatable_layers.append({"name": name, "type": "bias"})
            elif kind == "innerProduct":
                if layer.innerProduct.hasBias:
                    updatable_layers.append({"name": name, "type": "bias"})
            else:
                raise ValueError(f"Unexpected updatable layer {layer} of kind {kind}")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
            print(f"could not make {name} updatable")
    return updatable_layers


def save_builder(builder: NeuralNetworkBuilder, directory: str):
    save_spec(builder.spec, directory)
