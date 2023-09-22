from fed_kit import *

tflite_file = "mnist.tflite"
coreml_file = "mnist.mlmodel"
name = "mnist_unified"
tflite_layers = [1152, 128, 36864, 128, 1600000, 2000, 20000, 40]
coreml_layers = [
    {"name": "sequential/conv2d/BiasAdd", "type": "weights"},
    {"name": "sequential/conv2d/BiasAdd", "type": "bias"},
    {"name": "sequential/conv2d_1/BiasAdd", "type": "weights"},
    {"name": "sequential/conv2d_1/BiasAdd", "type": "bias"},
    {"name": "sequential/dense/BiasAdd", "type": "weights"},
    {"name": "sequential/dense/BiasAdd", "type": "bias"},
    {"name": "sequential/dense_1/BiasAdd", "type": "weights"},
    {"name": "sequential/dense_1/BiasAdd", "type": "bias"},
]
data_type = "MNIST_28x28x1"
response = upload(
    tflite_file, coreml_file, name, tflite_layers, coreml_layers, data_type
)
if response.status_code < 200 or response.status_code >= 300:
    print(response.text)
    exit(1)
print("Successfully uploaded the unified MNIST model.")
