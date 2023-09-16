import tensorflow as tf
from coremltools.models import datatypes
from coremltools.models.neural_network import AdamParams, NeuralNetworkBuilder

from .. import convert, nn_builder, random_fit, save_builder, try_make_layers_updatable

k = tf.keras
in_shape = (28, 28, 1)
n_classes = 10
file_name = "mnist.mlmodel"


def conv_layer():
    return k.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        input_shape=in_shape,
    )


def pool_layer():
    return k.layers.MaxPool2D((2, 2), strides=(2, 2))


def mnist_model():
    model = k.Sequential(
        [
            conv_layer(),
            pool_layer(),
            conv_layer(),
            pool_layer(),
            k.layers.Flatten(),
            k.layers.Dense(500, activation="relu", kernel_initializer="he_uniform"),
            k.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def config_builder(builder: NeuralNetworkBuilder):
    builder.set_mean_squared_error_loss(
        "lossLayer",
        input_feature=("sequential/dense_1/BiasAdd", datatypes.Array(n_classes)),
    )
    builder.set_adam_optimizer(AdamParams())
    builder.set_epochs(10)


def main():
    model = mnist_model()
    random_fit(model, in_shape)
    model.summary()
    mlmodel = convert(model)
    builder = nn_builder(mlmodel)
    config_builder(builder)
    updatables = try_make_layers_updatable(builder, 2)
    builder.inspect_layers()
    save_builder(builder, file_name)
    print(f"Updatable layers:\n{updatables}")
