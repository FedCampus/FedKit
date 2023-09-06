from pickle import loads

from django.db import models
from numpy.typing import NDArray


class TrainingDataType(models.Model):
    name = models.CharField(max_length=256, unique=True, null=False, editable=False)


# Always change together with `serializers.TFLiteModelSerializer`
# & Android `db.TFLiteModel`
# & Flutter `ml_models.TFliteModel`.
class TFLiteModel(models.Model):
    name = models.CharField(max_length=64, unique=True, null=False, editable=False)
    file_path = models.CharField(max_length=64, unique=True, null=False, editable=False)
    layers_sizes = models.JSONField(null=False, editable=False)
    """For TFLite, size of each layer of parameters in bytes.
    For CoreML, {name: str, shape: [int]} of each layer of parameters."""
    data_type = models.ForeignKey(
        TrainingDataType,
        on_delete=models.CASCADE,
        related_name="tflite_models",
        null=False,
        editable=False,
    )
    is_coreml = models.BooleanField(null=False, editable=False, default=False)

    def __str__(self) -> str:
        return f"TFLiteModel {self.name} for {self.data_type.name} at \
{self.file_path}, {len(self.layers_sizes)} layers"


class ModelParams(models.Model):
    params = models.BinaryField(null=False, editable=False)
    tflite_model = models.ForeignKey(
        TFLiteModel,
        on_delete=models.CASCADE,
        related_name="params",
        null=False,
        editable=False,
    )

    def decode_params(self) -> list[NDArray]:
        return loads(self.params)

    def __str__(self) -> str:
        return f"ModelParams for {self.tflite_model.name}: {self.decode_params()}"
