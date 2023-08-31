from train.models import *
from train.serializers import *

d = TrainingDataType(name="MNIST_28x28x1")
d.save()
file = "/static/MNIST--mnist.mlmodel"
name = "MNIST"
m = TFLiteModel(name=name, file_path=file, layers_sizes=[], data_type=d, is_coreml=True)
m.save()
s = TFLiteModelSerializer(m)
assert s.data["name"] == name
assert s.data["file_path"] == file
print("Successfully added CoreML MNIST data type and model to the database.")
