from .. import tf

k = tf.keras
in_shape = (28, 28, 1)


def conv_layer():
    return k.layers.Conv2D(32, (3, 3), activation="relu")


def pool_layer():
    return k.layers.MaxPool2D((2, 2))


def mnist_model():
    model = k.Sequential(
        [
            k.Input(shape=in_shape),
            k.layers.Flatten(),
            k.layers.Dense(512, activation="relu", kernel_initializer="he_uniform"),
            k.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model
