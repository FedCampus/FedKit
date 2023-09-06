import logging
from typing import IO, OrderedDict

from django.core.files.uploadedfile import UploadedFile
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.request import MultiValueDict # type: ignore
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from rest_framework.views import Request # type: ignore
from train import scheduler
from train.models import ModelParams, TFLiteModel, TrainingDataType
from train.scheduler import server
from train.serializers import (
    PostAdvertisedDataSerializer,
    PostServerDataSerializer,
    TFLiteModelSerializer,
    UploadTFLiteSerializer,
)

from backend.settings import BASE_DIR

logger = logging.getLogger(__name__)


def deserialize(cls, data):
    """Deserialize `data` using `cls`.
    Return `(validated_data, err)`."""
    serializer = cls(data=data)
    is_valid = serializer.is_valid()
    validated: OrderedDict = serializer.validated_data  # type: ignore
    if is_valid:
        return (validated, None)
    else:
        logger.error(serializer.errors)
        return (validated, serializer.errors)


def tflite_model_for_data_type(data: OrderedDict):
    try:
        data_type = TrainingDataType.objects.get(name=data["data_type"])
        filter = TFLiteModel.objects.filter(
            data_type=data_type, is_coreml=data["is_coreml"]
        )
        return filter.last()
    except Exception as err:
        logger.error(f"{err} while looking up model for `{data}`.")
        return


# https://www.django-rest-framework.org/api-guide/views/#api_view
@api_view(["POST"])
# https://stackoverflow.com/questions/31335736/cannot-apply-djangomodelpermissions-on-a-view-that-does-not-have-queryset-pro
@permission_classes((permissions.AllowAny,))
def advertise_model(request: Request):
    (data, err) = deserialize(PostAdvertisedDataSerializer, request.data)
    if err:
        return Response(err, HTTP_400_BAD_REQUEST)
    model = tflite_model_for_data_type(data)
    if model is None:
        return Response("No model corresponding to data_type", HTTP_404_NOT_FOUND)
    serializer = TFLiteModelSerializer(model)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def request_server(request: Request):
    (data, err) = deserialize(PostServerDataSerializer, request.data)
    if err:
        return Response(err, HTTP_400_BAD_REQUEST)
    try:
        model = TFLiteModel.objects.get(pk=data["id"])
    except TFLiteModel.DoesNotExist:
        logger.error(f"Model with id {data['id']} not found.")
        return Response("Model not found", HTTP_404_NOT_FOUND)
    response = server(model, data["start_fresh"])
    return Response(response.__dict__)


def file_in_request(request: Request):
    files = request.FILES
    if isinstance(files, MultiValueDict):
        file = files.get("file")
        if isinstance(file, UploadedFile) and file.name and file.file:
            return (file.name, file.file)


def tflite_name_not_unique(file_name: str):
    try:
        _ = TFLiteModel.objects.get(name=file_name)
        return True
    except TFLiteModel.DoesNotExist:
        return False


def get_data_type(data_type_name: str):
    try:
        data_type = TrainingDataType.objects.get(name=data_type_name)
    except TrainingDataType.DoesNotExist:
        logger.warn(f"upload: Creating new data_type `{data_type_name}`.")
        data_type = TrainingDataType(name=data_type_name)
        data_type.save()
    return data_type


def save_model_file(name: str, file_name: str, file: IO):
    """Given that the name is unique, guarantee unique file name."""
    path = f"static/{name}--{file_name}"
    with open(BASE_DIR / path, "wb") as fd:
        fd.write(file.read())
    return path


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def upload_tflite(request: Request):
    (data, err) = deserialize(UploadTFLiteSerializer, request.data)
    if err:
        return Response(err, HTTP_400_BAD_REQUEST)
    name = data["name"]
    data_type_name = data["data_type"]
    if tflite_name_not_unique(name):
        return Response("Model name used", HTTP_400_BAD_REQUEST)
    file = file_in_request(request)
    if file is None:
        return Response("No file in request.", HTTP_400_BAD_REQUEST)
    data_type = get_data_type(data_type_name)
    path = save_model_file(name, *file)
    model = TFLiteModel(
        name=name,
        file_path=f"/{path}",
        layers_sizes=data["layers_sizes"],
        data_type=data_type,
    )
    model.save()

    return Response("ok")


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def store_params(request: Request):
    # TODO: CoreML server.
    server = scheduler.tf_server
    if server is None:
        logger.error("No server running but got params to store.")
        return Response("No server running.", HTTP_400_BAD_REQUEST)
    file = file_in_request(request)
    if file is None:
        return Response("No file in request.", HTTP_400_BAD_REQUEST)
    params = file[1].read()
    to_save = ModelParams(params=params, tflite_model=server.model)
    to_save.save()
    server.update_session_end_time()
    return Response("ok")
