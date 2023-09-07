from pickle import loads

from django.db import models
from numpy.typing import NDArray


class TrainingDataType(models.Model):
    name = models.CharField(max_length=256, unique=True, editable=False)

    def __str__(self):
        return self.name


# Always change together with `serializers.MLModelSerializer`
# & Android `db.MLModel`
# & Flutter `ml_model.MLModel`.
class MLModel(models.Model):
    name = models.CharField(max_length=64, unique=True, editable=False)
    tflite_path = models.CharField(max_length=64, unique=True, null=True, default=None)
    """Path to `.tflite` file."""
    coreml_path = models.CharField(max_length=64, unique=True, null=True, default=None)
    """Path to `.mlmodel` file."""
    tflite_layers = models.JSONField(null=True, default=None)
    """Size of each layer of parameters in bytes."""
    coreml_layers = models.JSONField(null=True, default=None)
    """`[{name, type}]` of each layer of parameters.
    `type` can either be `"weights"` or `"bias"`."""
    data_type = models.ForeignKey(
        TrainingDataType,
        on_delete=models.CASCADE,
        related_name="ml_models",
        editable=False,
    )
    tflite = models.BooleanField(default=True)
    coreml = models.BooleanField(default=False)

    def __str__(self) -> str:
        desc = [f"MLModel {self.name} for {self.data_type.name}"]
        if self.tflite:
            desc.append(
                f"TFLite model at {self.tflite_path} of \
{len(self.tflite_layers)} layers"
            )
        if self.coreml:
            desc.append(
                f"CoreML model at {self.coreml_path} of \
{len(self.coreml_layers)} layers"
            )
        return ", ".join(desc)


class ModelParams(models.Model):
    params = models.BinaryField(editable=False)
    tflite_model = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name="params",
        editable=False,
    )

    def decode_params(self) -> list[NDArray]:
        return loads(self.params)

    def __str__(self) -> str:
        return f"ModelParams for {self.tflite_model.name}: {self.decode_params()}"
