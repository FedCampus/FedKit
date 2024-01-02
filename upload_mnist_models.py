from fed_kit import *

tflite_file = "mnist.tflite"
coreml_file = "mnist.mlmodel"
name = "mnist_unified"
tflite_layers = [401408, 512, 256000, 2000, 20000, 40]
coreml_layers = [
    {"name": "sequential/dense/BiasAdd", "type": "weights", "updatable": True},
    {"name": "sequential/dense/BiasAdd", "type": "bias", "updatable": True},
    {"name": "sequential/dense_1/BiasAdd", "type": "weights", "updatable": True},
    {"name": "sequential/dense_1/BiasAdd", "type": "bias", "updatable": True},
    {"name": "Identity", "type": "weights", "updatable": True},
    {"name": "Identity", "type": "bias", "updatable": True},
]
data_type = "MNIST_28x28x1"
response = upload(
    tflite_file, coreml_file, name, tflite_layers, coreml_layers, data_type
)
if response.status_code < 200 or response.status_code >= 300:
    print(response.text)
    exit(1)
print("Successfully uploaded the unified MNIST model.")
