# Generate TFLite model files

This module provides infrastructure to generate `.tflite` files that are compatible with the Android package.

For example implementations, check out `cifar10_eg/` and `toy_regression_eg/`.

## Dependencies installation

```sh
python3 -m pip install -r requirements.txt
```

Please ignore `./pyproject.toml`.

## Model declaration

Inherit from `BaseTFLiteModel`, annotate with `@tflite_model_class`, and override `X_SHAPE`, `Y_SHAPE`, and `__init__` to assign `self.model`.
For example:

```python
@tflite_model_class
class MyModel(BaseTFLiteModel):
    X_SHAPE = [WIDTH, HEIGHT]
    Y_SHAPE = [N_CLASSES]

    def __init__(self):
        self.model = tf.keras.Sequential([
            # Layers.
        ])
        self.model.compile()
```

If you are not content with the default implementation of `train`, `infer`, `parameters`, `restore` provided by `BaseTFLiteModel`, override them.
If you want the change how the `tf.function` conversions are done, though, you would need to write the whole class yourself (exactly like below):

```python
class CustomModel(tf.Module):
    X_SHAPE = …
    Y_SHAPE = …

    def __init__(self, …):
        self.model = …
        self.model.compile()
        # …

    @tf.function(input_signature=[
        # `TensorSpec` of `x` and `y`.
    ])
    def train(self, x, y):
        return {"loss": …}

    @tf.function(input_signature=[
        # `TensorSpec` of `y`.
    ])
    def infer(self, x):
        return {"logits": …}

    @tf.function(input_signature=[])
    def parameters(self):
        return {"a0": …, "a1": …, …}

    @tf.function()
    def restore(self, **parameters):
        # …
        return self.parameters()
```

## TFLite file generation

```python
model = MyModel()
save_model(model, SAVED_MODEL_DIR)
tflite_model = convert_saved_model(SAVED_MODEL_DIR)
save_tflite_model(tflite_model, "my_model.tflite")
```

The script prints the `Model parameter sizes in bytes` in <span style="color: red;">red</span>. Copy that red list for later storing into the database.
That list is `TFLiteModel.layers_sizes` for your model.

The above script generates the `.tflite` file at `../my_model.tflite`. Move that file to `../backend/static/a_more_proper_name.tflite`.

Go to `../backend` and add the model into your database.

To suppress Tensorflow printing, try:

```sh
export TF_CPP_MIN_LOG_LEVEL=2
```

For validation, see the `toy_regression_eg/` example.

## Running demo

Please see `cifar10_eg/README.md` and `toy_regression_eg/README.md`.
