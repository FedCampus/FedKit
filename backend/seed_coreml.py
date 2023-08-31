from train.models import *
from train.serializers import *

d = TrainingDataType(name="MNIST_28x28x1")
d.save()
file = "/static/MNIST--mnist.mlmodel"
name = "MNIST"
layers = [
    {"name": "conv1", "shape": [32, 1, 3, 3]},
    {"name": "conv2", "shape": [32, 32, 2, 2]},
    {"name": "hidden1", "shape": [500, 1152]},
    {"name": "hidden2", "shape": [10, 500]},
]
m = TFLiteModel(
    name=name, file_path=file, layers_sizes=layers, data_type=d, is_coreml=True
)
m.save()
s = TFLiteModelSerializer(m)
assert s.data["name"] == name
assert s.data["file_path"] == file
assert s.data["layers_sizes"] == layers
print("Successfully added CoreML MNIST data type and model to the database.")
