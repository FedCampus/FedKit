from train.models import *
from train.serializers import *

d = TrainingDataType(name="CIFAR10_32x32x3")
d.save()
file = "/static/cifar10.tflite"
sizes = [1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40]
name = "CIFAR10"
m = TFLiteModel(name=name, file_path=file, layers_sizes=sizes, data_type=d)
m.save()
s = TFLiteModelSerializer(m)
assert s.data["name"] == name
assert s.data["file_path"] == file
assert s.data["layers_sizes"] == sizes
print("Successfully added CIFAR10 data type and model to the database.")
