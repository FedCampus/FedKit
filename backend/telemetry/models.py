from django.db import models
from train.models import TFLiteModel

cfg = {"null": False, "editable": False}


class TrainingSession(models.Model):
    id: int  # Help static analysis.
    tflite_model = models.ForeignKey(
        TFLiteModel, on_delete=models.CASCADE, related_name="training_sessions", **cfg
    )
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Training session {self.id} for {self.tflite_model} {self.start_time} - {self.end_time}>"


# Always change together with Android `Train.FitInsTelemetryData`.
class FitInsTelemetryData(models.Model):
    id: int  # Help static analysis.
    device_id = models.BigIntegerField(**cfg)
    session_id = models.ForeignKey(
        TrainingSession, on_delete=models.CASCADE, related_name="fit_ins", **cfg
    )
    start = models.DateTimeField(**cfg)
    end = models.DateTimeField(**cfg)

    def __str__(self) -> str:
        return f"FitIns {self.id} on {self.device_id} {self.start} - {self.end}"


# Always change together with Android `Train.EvaluateInsTelemetryData`.
class EvaluateInsTelemetryData(models.Model):
    id: int  # Help static analysis.
    device_id = models.BigIntegerField(**cfg)
    session_id = models.ForeignKey(
        TrainingSession, on_delete=models.CASCADE, related_name="evaluate_ins", **cfg
    )
    start = models.DateTimeField(**cfg)
    end = models.DateTimeField(**cfg)
    loss = models.FloatField(**cfg)
    accuracy = models.FloatField(**cfg)
    test_size = models.BigIntegerField(**cfg)

    def __str__(self) -> str:
        return f"EvaluateIns {self.id} on {self.device_id} {self.start} - {self.end} loss: {self.loss} accuracy: {self.accuracy} test_size: {self.test_size}"
