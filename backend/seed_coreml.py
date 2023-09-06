from train.models import MLModel, TrainingDataType
from train.serializers import MLModelSerializer

d = TrainingDataType(name="MNIST_28x28x1")
d.save()
file = "/static/MNIST--mnist.mlmodel"
name = "MNIST"
layers = [
    {"name": "conv1", "type": "weights"},
    {"name": "conv2", "type": "weights"},
    {"name": "hidden1", "type": "weights"},
    {"name": "hidden2", "type": "weights"},
]
m = MLModel(
    name=name,
    coreml_path=file,
    coreml_layers=layers,
    data_type=d,
    tflite=False,
    coreml=True,
)
m.save()
s = MLModelSerializer(m)
assert s.data["name"] == name
assert s.data["coreml_path"] == file
assert s.data["coreml_layers"] == layers
assert s.data["coreml"]
print("Successfully added CoreML MNIST data type and model to the database.")
