# FedKit

Federated Learning (FL) development kit for FedCampus.

> A FatKid to support the FatCampus.

This repository contains the libraries used to implement the FL functionalities of the FedCampus app.

## Architecture

### Backend

The backend server uses Django REST Framework to provide persistent service and on-demand, backend-configurable training.

The FL training steps are supported using the [Flower][flower] FL framework. Flower servers use gRPC, and are spawned on-demand by the backend server.

Located at `backend/`. Please see `backend/README.md` for information on how to set up the backend server.

## Mobile clients

### Android client

The Android client package and demo is located at `client/`.

To try out the Android client, please see `client/README.md`.

To use the library in this repository to implement custom Android Flower clients, please see `client/fedcampus/README.md`.

### Flutter Android & iOS client

The Flutter client package to communicate with the backend and Flower servers is at `flutter/`.

The example Flutter client for both Android and iOS is at `fed_kit_client/`. To try it out, please see `fed_kit_client/README.md`.

## ML model generation

We support TensorFlow (Keras) models for Android and Core ML models for iOS.

### ML model for Android

The ML model generation script for Android is located at `gen_tflite/`. Please see `gen_tflite/README.md` for information on how to create models and convert them to `.tflite` files.

### ML model for iOS

`gen_mlmodel` is still a work in progress. Meanwhile, you can construct Core ML models manually using `coremltools`.

## Training procedure

1. Client asks backend what which model to use based on its `data_type`.
1. Client downloads that model if it does not have it.
1. Client asks backend for a Flower server to train with that model.

## Development

### Development on Python code

- Use Python3.10 or above.
- Install dependencies using `requirements.txt`s.

## Contributing

Please see [CONTRIBUTING.md for FedCampus][contributing].

## History

This repository is moved from [dyn_flower_android_drf][dyn_flower_android_drf] to [FedKit][fed_kit]. Please see the older Git history in the former repository.

[contributing]: https://github.com/FedCampus/meta/blob/main/CONTRIBUTING.md
[dyn_flower_android_drf]: https://github.com/FedCampus/dyn_flower_android_drf
[fed_kit]: https://github.com/FedCampus/FedKit
[flower]: https://flower.dev/
