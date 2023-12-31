from .. import tf

k = tf.keras
in_shape = (28, 28, 1)
n_classes = 10


def conv_layer():
    return k.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
    )


def pool_layer():
    return k.layers.MaxPool2D((2, 2), strides=(2, 2))


def mnist_model():
    model = k.Sequential(
        [
            k.Input(shape=in_shape),
            conv_layer(),
            pool_layer(),
            conv_layer(),
            pool_layer(),
            k.layers.Flatten(),
            k.layers.Dense(500, activation="relu", kernel_initializer="he_uniform"),
            k.layers.Dense(n_classes),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model
