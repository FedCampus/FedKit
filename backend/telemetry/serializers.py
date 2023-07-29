from datetime import datetime

from rest_framework import serializers
from telemetry.models import *


class TimestampMillis(serializers.Field):
    def to_internal_value(self, milliseconds: int) -> datetime:
        return datetime.fromtimestamp(milliseconds / 1000)

    def to_representation(self, timestamp: datetime) -> int:
        return round(timestamp.timestamp() * 1000)


class FitInsTelemetryDataSerializer(serializers.Serializer):
    device_id = serializers.IntegerField()
    session_id = serializers.IntegerField()
    start = TimestampMillis()
    end = TimestampMillis()

    def create(self, validated_data):
        session_id = validated_data["session_id"]
        validated_data["session_id"] = TrainingSession.objects.get(id=session_id)
        return FitInsTelemetryData.objects.create(**validated_data)

    class Meta:
        model = FitInsTelemetryData
        fields = ["device_id", "session_id", "start", "end"]


class EvaluateInsTelemetryDataSerializer(serializers.Serializer):
    device_id = serializers.IntegerField()
    session_id = serializers.IntegerField()
    start = TimestampMillis()
    end = TimestampMillis()
    loss = serializers.FloatField()
    accuracy = serializers.FloatField()
    test_size = serializers.IntegerField()

    def create(self, validated_data):
        session_id = validated_data["session_id"]
        validated_data["session_id"] = TrainingSession.objects.get(id=session_id)
        return EvaluateInsTelemetryData.objects.create(**validated_data)

    class Meta:
        model = EvaluateInsTelemetryData
        fields = [
            "device_id",
            "session_id",
            "start",
            "end",
            "loss",
            "accuracy",
            "test_size",
        ]
