# FedKit

Federated Learning (FL) and Federated Analytics (FA) development kit for FedCampus.

This repository contains the libraries used to implement the FL and FA functionalities of the FedCampus app.

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

### iOS client

- [ ] Training using Flower.
- [ ] Communication with the Backend.

## ML model generation

We support TensorFlow (Keras) models.

### ML model for Android

The ML model generation script for Android is located at `gen_tflite/`. Please see `gen_tflite/README.md` for information on how to create models and convert them to `.tflite` files.

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
