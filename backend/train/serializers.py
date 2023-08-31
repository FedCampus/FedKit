from rest_framework import serializers
from train.models import TFLiteModel


# Always change together with `models.TFLiteModel`.
class TFLiteModelSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    file_path = serializers.CharField()
    layers_sizes = serializers.ListField(child=serializers.IntegerField(min_value=0))
    is_coreml = serializers.BooleanField()

    class Meta:
        model = TFLiteModel
        fields = ["id", "name", "file_path", "layers_sizes", "is_coreml"]


# Always change together with Dart `backend_client.PostAdvertisedData`.
class PostAdvertisedDataSerializer(serializers.Serializer):
    data_type = serializers.CharField(max_length=256)
    is_coreml = serializers.BooleanField(required=False, default=False)  # type: ignore


# Always change together with Android `HttpClient.PostServerData`
# & Dart `backend_client.PostServerData`.
class PostServerDataSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    start_fresh = serializers.BooleanField(required=False, default=False)  # type: ignore


# Always change together with `upload` in `fed_kit.py`.
class UploadTFLiteSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=256)
    layers_sizes = serializers.ListField(child=serializers.IntegerField(min_value=0))
    data_type = serializers.CharField(max_length=256)
    is_coreml = serializers.BooleanField(required=False, default=False)  # type: ignore
