# Android Client

## Set up

Download the training and testing data from <https://www.dropbox.com/s/coeixr4kh8ljw6o/cifar10.zip?dl=1> and extract them to `app/src/main/assets/data`.

## Run the demo

Set up and start the backend server according to `Set up` and `Development` sections in `../backend/README.md`.

Install `app` on *physical* Android devices and launch it.

*Note*: the highest tested JDK version the app supports is 16; it fails to build using JDK 19 on macOS.

In the user interface, fill in:

- Device number: a unique number among 1 ~ 10.
    This number is used to choose the partition of the training dataset.
- Server IP: an IPv4 address of the computer your backend server is running on. You can probably find it in your system network settings.
- Server port: 8000, if you follow the `Development` section mentioned above.

Push the second button and connect to the backend server. This should take little time.

Push the first button and load the dataset. This may take a minute.

Push the last button and start the training.

## Credits

`app` is developed based on [Flower Android Java example][flower_java]. The related portion of the code was licensed under [Flower's license][flower_license].

## Note on other code in this directory

This directory also contains the Android library code used to implement `app` and a benchmark app.

[flower_java]: https://github.com/adap/flower/tree/main/examples/android
[flower_license]: https://github.com/adap/flower/blob/main/LICENSE
