from pickle import loads

from django.db import models
from numpy.typing import NDArray


class TrainingDataType(models.Model):
    name = models.CharField(max_length=256, unique=True, null=False, editable=False)


# Always change together with `serializers.TFLiteModelSerializer`
# & Android `db.TFLiteModel`
# & Flutter `ml_models.TFliteModel`.
class MLModel(models.Model):
    name = models.CharField(max_length=64, unique=True, null=False, editable=False)
    tflite_path = models.CharField(max_length=64, unique=True, default=None)
    """Path to `.tflite` file."""
    coreml_path = models.CharField(max_length=64, unique=True, default=None)
    """Path to `.mlmodel` file."""
    tflite_layers = models.JSONField(default=None)
    """Size of each layer of parameters in bytes."""
    coreml_layers = models.JSONField(default=None)
    """`[name: [types]]` of each layer of parameters.
    `type` can either be `"weights"` or `"bias"`."""
    data_type = models.ForeignKey(
        TrainingDataType,
        on_delete=models.CASCADE,
        related_name="ml_models",
        null=False,
        editable=False,
    )
    tflite = models.BooleanField(null=False, default=True)
    coreml = models.BooleanField(null=False, default=False)

    def __repl__(self) -> str:
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
    params = models.BinaryField(null=False, editable=False)
    tflite_model = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name="params",
        null=False,
        editable=False,
    )

    def decode_params(self) -> list[NDArray]:
        return loads(self.params)

    def __str__(self) -> str:
        return f"ModelParams for {self.tflite_model.name}: {self.decode_params()}"
