from train.models import MLModel, TrainingDataType
from train.serializers import MLModelSerializer

d = TrainingDataType(name="CIFAR10_32x32x3")
d.save()
file = "/static/cifar10.tflite"
sizes = [1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40]
name = "CIFAR10"
m = MLModel(name=name, tflite_path=file, tflite_layers=sizes, data_type=d)
m.save()
s = MLModelSerializer(m)
assert s.data["name"] == name
assert s.data["tflite_path"] == file
assert s.data["tflite_layers"] == sizes
print("Successfully added CIFAR10 data type and model to the database.")
