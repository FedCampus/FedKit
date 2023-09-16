from rest_framework import serializers
from train.models import MLModel


# Always change together with `models.MLModel`.
class MLModelSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    tflite_path = serializers.CharField(allow_null=True)
    coreml_path = serializers.CharField(allow_null=True)
    tflite_layers = serializers.JSONField(allow_null=True)
    coreml_layers = serializers.JSONField(allow_null=True)
    tflite = serializers.BooleanField()
    coreml = serializers.BooleanField()

    class Meta:
        model = MLModel
        fields = [
            "id",
            "name",
            "tflite_path",
            "coreml_path",
            "tflite_layers",
            "coreml_layers",
            "tflite",
            "coreml",
        ]


# Always change together with Dart `backend_client.PostAdvertisedData`.
class PostAdvertisedDataSerializer(serializers.Serializer):
    data_type = serializers.CharField(max_length=256)
    tflite = serializers.BooleanField(required=False, default=True)
    coreml = serializers.BooleanField(required=False, default=False)


# Always change together with Android `HttpClient.PostServerData`
# & Dart `backend_client.PostServerData`.
class PostServerDataSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    start_fresh = serializers.BooleanField(required=False, default=False)


# Always change together with `upload` in `fed_kit.py`.
class UploadModelSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=256)
    tflite_layers = serializers.ListField(
        allow_null=True, child=serializers.IntegerField(min_value=0)
    )
    coreml_layers = serializers.ListField(
        allow_null=True, child=serializers.JSONField()
    )
    data_type = serializers.CharField(max_length=256)
