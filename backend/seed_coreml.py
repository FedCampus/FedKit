from train.models import *
from train.serializers import *

d = TrainingDataType(name="MNIST_28x28x1")
d.save()
file = "/static/MNIST--mnist.mlmodel"
names = ["sequential/conv2d/Conv2Dx", "sequential/conv2d_1/Conv2Dx"]
name = "MNIST"
m = CoreMLModel(name=name, file_path=file, layers_names=names, data_type=d)
m.save()
s = CoreMLModelSerializer(m)
assert s.data["name"] == name
assert s.data["file_path"] == file
assert s.data["layers_names"] == names
print("Successfully added CoreML MNIST data type and model to the database.")
