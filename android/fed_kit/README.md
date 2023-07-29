# FedCampus Android Package

This package can be used to implement custom Flower clients.
It provides the class `Train` for training and tracking training-related data.

Please check out `../app/` for an example usage.

## General process

To use `Train` for a custom Flower client, follow the following steps:

### Data type preparation

Decide the input and output types for training.
They also need to match the TFLite model file used.

1. Decide `dataType: String`, the tag for training input type.
    For example, we named the training input for CIFAR10 `"CIFAR10_32x32x3"`.
1. Decide `X`, the training input type.
    `X` must be an array type that TFLite interpreter takes,
    such as `FloatArray`, `Array<FloatArray>`, `DoubleArray`,
    or any nested versions.
    It cannot be buffer types such as `ByteBuffer` because the interpreter
    would not be able to handle dynamic input and output shape used to run with
    batches of data.
1. Decide `Y`, the training output type.
    Usually, it is simply `FloatArray`.

### Data structure initialization

1. In your view, add `train` as an attribute, with the concrete `X` and `Y`.
    For example:

    ```kotlin
    lateinit var train: Train<Float3DArray, FloatArray>`
    ```

1. Create `sampleSpec: SampleSpec<X, Y>`. For example:

    ```kotlin
    val sampleSpec = SampleSpec<Array<Array<FloatArray>>, FloatArray>(
        { it.toTypedArray() },
        { it.toTypedArray() },
        { Array(it) { FloatArray(CLASSES.size) } },
        ::negativeLogLikelihoodLoss,
        ::classifierAccuracy,
    )
    ```

    The first 2 arguments would always be `{ it.toTypedArray() }`.
    They exist only because the languages' limitations on generics.

    The 3rd argument is a closure that produces an array of empty `Y`s.

    The 4th and 5th arguments are the loss function and the accuracy function.

1. Initialize `Train<X, Y>` somewhere after the view is initialized.

    ```kotlin
    train = Train(this, backendUrl, sampleSpec)
    ```

### Model acquisition and server-side training initiation

In sequence, call these methods on `train`:

1. `prepareModel` with your `dataType` to obtain the model from
    backend server.
1. `getServerInfo` to ask the backend server to initialize a Flower
    server for training.
1. `prepare` to initialize `train.flowerClient` and establish connection to
    the Flower server.

### Data loading

Load data into `train.flowerClient` using `flowerClient.addSample`.

### Start training

Call `train.start`.
