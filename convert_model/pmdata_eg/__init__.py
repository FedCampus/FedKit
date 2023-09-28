# from .. import tf
import tensorflow as tf

k = tf.keras


def build_model():
    model = k.Sequential()
    # activation=hp.Choice("activation", ["relu", "tanh", 'linear'])
    # Tune the number of layers.
    for i in range(2):
        model.add(
            k.layers.Dense(
                # Tune number of units separately.
                units=512,
                activation="relu",
                ## TODO: hard code the feature
                input_dim=7,
            )
        )

    model.add(k.layers.Dense(1))
    # learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    learning_rate = 0.0013826
    # optimizer=hp.Choice(name="optimizer", values=["rmsprop", "adam", "adadelta"])
    model.compile(
        # optimizer=optimizer,
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


if __name__ == "__main__":
    ## test code
    import pandas as pd

    df = pd.read_csv(r"merged_features_efficiency_FINAL2_modified_Beilong.csv")
    data = df[
        [
            "steps",
            "calories",
            "distance_km",
            "very_active_minutes",
            "moderately_active_minutes",
            "lightly_active_minutes",
            "sedentary_minutes",
            "efficiency",
        ]
    ]
    # one-hot encode the values in the gender column with pd.get_dummies.
    # data = pd.get_dummies(data, columns=["gender"], prefix="", prefix_sep="")
    # shuffle the data. I noticed that suffling the data increased the loss compared with non-shuffled data
    data = data.sample(frac=1).reset_index(drop=True)

    # f_columns=['age','height','steps','calories','very_active_minutes','moderately_active_minutes','lightly_active_minutes','sedentary_minutes','female','male']
    f_columns = [
        "steps",
        "distance_km",
        "calories",
        "very_active_minutes",
        "moderately_active_minutes",
        "lightly_active_minutes",
        "sedentary_minutes",
    ]
    n = len(data)
    num_train_samples = int(n * 0.7)
    num_val_samples = int(n * 0.9) - int(n * 0.7)
    num_test_samples = n - int(n * 0.9)

    train_data = data[0 : int(n * 0.7)]
    val_data = data[int(n * 0.7) : int(n * 0.9)]
    test_data = data[int(n * 0.9) :]

    train_features = train_data[f_columns]
    val_features = val_data[f_columns]
    test_features = test_data[f_columns]

    train_target = train_data["efficiency"]
    val_target = val_data["efficiency"]
    test_target = test_data["efficiency"]

    model = build_model()
    hist = model.fit(
        train_features,
        train_target,
        epochs=300,
        callbacks=k.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        verbose=1,
        validation_data=(val_features, val_target),
    )
    pass
