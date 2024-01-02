from .. import tf

k = tf.keras
in_shape = (28, 28, 1)
n_classes = 10


def mnist_model():
    model = k.Sequential(
        [
            k.Input(shape=in_shape),
            k.layers.Flatten(),
            k.layers.Dense(128, activation="relu"),
            k.layers.Dense(500, activation="relu"),
            k.layers.Dense(n_classes),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model
