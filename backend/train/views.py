import logging
from typing import IO, OrderedDict

from django.core.files.uploadedfile import UploadedFile
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.request import MultiValueDict  # type: ignore
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from rest_framework.views import Request  # type: ignore
from train import scheduler
from train.models import MLModel, ModelParams, TrainingDataType
from train.scheduler import server
from train.serializers import (
    MLModelSerializer,
    PostAdvertisedDataSerializer,
    PostServerDataSerializer,
    UploadModelSerializer,
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


def ml_model_for_data_type(data: OrderedDict):
    try:
        data_type = TrainingDataType.objects.get(name=data["data_type"])
        filter = MLModel.objects.filter(data_type=data_type)
        if data["tflite"]:
            filter = filter.filter(tflite=True)
        if data["coreml"]:
            filter = filter.filter(coreml=True)
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
    model = ml_model_for_data_type(data)
    if model is None:
        return Response("No model corresponding to data_type", HTTP_404_NOT_FOUND)
    serializer = MLModelSerializer(model)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def request_server(request: Request):
    (data, err) = deserialize(PostServerDataSerializer, request.data)
    if err:
        return Response(err, HTTP_400_BAD_REQUEST)
    try:
        model = MLModel.objects.get(pk=data["id"])
    except MLModel.DoesNotExist:
        logger.error(f"Model with id {data['id']} not found.")
        return Response("Model not found", HTTP_404_NOT_FOUND)
    response = server(model, data["start_fresh"])
    return Response(response.__dict__)


def file_in_request(request: Request, name: str):
    files = request.FILES
    if isinstance(files, MultiValueDict):
        file = files.get(name)
        if isinstance(file, UploadedFile) and file.name and file.file:
            return (file.name, file.file)


def model_name_not_unique(file_name: str):
    try:
        _ = MLModel.objects.get(name=file_name)
        return True
    except MLModel.DoesNotExist:
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
def upload_model(request: Request):
    (data, err) = deserialize(UploadModelSerializer, request.data)
    if err:
        return Response(err, HTTP_400_BAD_REQUEST)
    name, data_type_name = data["name"], data["data_type"]
    if model_name_not_unique(name):
        return Response("Model name used", HTTP_400_BAD_REQUEST)
    tflite = data["tflite_layers"] is not None
    coreml = data["coreml_layers"] is not None
    tflite_path, coreml_path = None, None
    if tflite:
        file = file_in_request(request, "tflite")
        if file is None:
            return Response("No TFLite file in request.", HTTP_400_BAD_REQUEST)
        tflite_path = save_model_file(name, *file)
    if coreml:
        file = file_in_request(request, "coreml")
        if file is None:
            return Response("No CoreML file in request.", HTTP_400_BAD_REQUEST)
        coreml_path = save_model_file(name, *file)
    data_type = get_data_type(data_type_name)
    model = MLModel(
        name=name,
        tflite_path=tflite_path,
        coreml_path=coreml_path,
        tflite_layers=data["tflite_layers"],
        coreml_layers=data["coreml_layers"],
        data_type=data_type,
        tflite=tflite,
        coreml=coreml,
    )
    model.save()

    return Response("ok")


# Always change together with `run.FedAvgAndroidSave`.
@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def store_params(request: Request):
    coreml = request.data.get("coreml", False)
    server = scheduler.cm_server if coreml else scheduler.tf_server
    if server is None:
        logger.error("No server running but got params to store.")
        return Response("No server running.", HTTP_400_BAD_REQUEST)
    file = file_in_request(request, "file")
    if file is None:
        return Response("No file in request.", HTTP_400_BAD_REQUEST)
    params = file[1].read()
    to_save = ModelParams(params=params, tflite_model=server.model)
    to_save.save()
    server.update_session_end_time()
    return Response("ok")
