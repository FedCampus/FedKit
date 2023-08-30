from rest_framework import serializers
from train.models import CoreMLModel, TFLiteModel


class TFLiteModelSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    file_path = serializers.CharField()
    layers_sizes = serializers.ListField(child=serializers.IntegerField(min_value=0))

    class Meta:
        model = TFLiteModel
        fields = ["id", "name", "file_path", "layers_sizes"]


class CoreMLModelSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    file_path = serializers.CharField()
    layers_names = serializers.ListField(child=serializers.CharField())

    class Meta:
        model = CoreMLModel
        fields = ["id", "name", "file_path", "layers_names"]


class PostAdvertisedDataSerializer(serializers.Serializer):
    data_type = serializers.CharField(max_length=256)


# Always change together with Android `HttpClient.PostServerData`
# & Dart `backend_client.PostServerData`.
class PostServerDataSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    start_fresh = serializers.BooleanField(required=False, default=False)  # type: ignore
    is_coreml = serializers.BooleanField(required=False, default=False)  # type: ignore


# Always change together with `upload` in `fed_kit.py`.
class UploadTFLiteSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=256)
    layers_sizes = serializers.ListField(child=serializers.IntegerField(min_value=0))
    data_type = serializers.CharField(max_length=256)


class UploadCoreMLSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=256)
    layers_names = serializers.ListField(max_length=256)
    data_type = serializers.CharField(max_length=256)
