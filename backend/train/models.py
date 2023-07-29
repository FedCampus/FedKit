from pickle import dumps, loads

from django.db import models
from numpy.typing import NDArray

cfg = {"null": False, "editable": False}


class TrainingDataType(models.Model):
    name = models.CharField(max_length=256, unique=True, **cfg)


# Always change together with Android `db.TFLiteModel.TFLiteModel`.
class TFLiteModel(models.Model):
    name = models.CharField(max_length=64, unique=True, **cfg)
    file_path = models.CharField(max_length=64, unique=True, **cfg)
    layers_sizes = models.JSONField(**cfg)
    """Size of each layer of parameters in bytes."""
    data_type = models.ForeignKey(
        TrainingDataType, on_delete=models.CASCADE, related_name="tflite_models", **cfg
    )

    def __str__(self) -> str:
        return f"TFLiteModel {self.name} for {self.data_type.name} at {self.file_path}, {len(self.layers_sizes)} layers"


class ModelParams(models.Model):
    """Do not initialize this class directly, use `make_model_params` instead."""

    params = models.BinaryField(**cfg)
    tflite_model = models.ForeignKey(
        TFLiteModel, on_delete=models.CASCADE, related_name="params", **cfg
    )

    def decode_params(self) -> list[NDArray]:
        return loads(self.params)

    def __str__(self) -> str:
        return f"ModelParams for {self.tflite_model.name}: {self.decode_params()}"


def make_model_params(params: list[NDArray], tflite_model: TFLiteModel) -> ModelParams:
    return ModelParams(params=dumps(params), tflite_model=tflite_model)
