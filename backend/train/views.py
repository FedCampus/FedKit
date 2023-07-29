import logging
from typing import OrderedDict

from django.core.files.uploadedfile import UploadedFile
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.request import MultiValueDict
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from rest_framework.views import Request
from train import scheduler
from train.models import *
from train.scheduler import server
from train.serializers import *

from backend.settings import BASE_DIR

logger = logging.getLogger(__name__)


def model_for_data_type(data_type):
    if not type(data_type) == str:
        logger.error(f"Looking up model for non-string data_type `{data_type}`.")
        return
    try:
        data_type = TrainingDataType.objects.get(name=data_type)
        return TFLiteModel.objects.filter(data_type=data_type).last()
    except Exception as err:
        logger.error(f"{err} while looking up model for data_type `{data_type}`.")
        return


# https://www.django-rest-framework.org/api-guide/views/#api_view
@api_view(["POST"])
# https://stackoverflow.com/questions/31335736/cannot-apply-djangomodelpermissions-on-a-view-that-does-not-have-queryset-pro
@permission_classes((permissions.AllowAny,))
def advertise_model(request):
    data_type = request.data.get("data_type")
    model = model_for_data_type(data_type)
    if model is None:
        return Response("No model corresponding to data_type", HTTP_404_NOT_FOUND)
    serializer = TFLiteModelSerializer(model)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def request_server(request: Request):
    serializer = PostServerDataSerializer(data=request.data)  # type: ignore
    if not serializer.is_valid():
        logger.error(serializer.errors)
        return Response(serializer.errors, HTTP_400_BAD_REQUEST)
    data: OrderedDict = serializer.validated_data  # type: ignore
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
        if isinstance(file, UploadedFile):
            return file


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def upload_file(request: Request):
    # Deserialize request data.
    serializer = UploadDataSerializer(data=request.data)  # type: ignore
    if not serializer.is_valid():
        logger.error(serializer.errors)
        return Response(serializer.errors, HTTP_400_BAD_REQUEST)
    data: OrderedDict = serializer.validated_data  # type: ignore
    name = data["name"]
    data_type_name = data["data_type"]
    # Validate unique file name.
    try:
        model = TFLiteModel.objects.get(name=data["name"])
        return Response("Model name used", HTTP_400_BAD_REQUEST)
    except TFLiteModel.DoesNotExist:
        pass
    # Get model file.
    file = file_in_request(request)
    if file is None:
        return Response("No file in request.", HTTP_400_BAD_REQUEST)
    # Get `data_type`.
    try:
        data_type = TrainingDataType.objects.get(name=data_type_name)
    except TrainingDataType.DoesNotExist:
        logger.warn(f"upload: Creating new data_type `{data_type_name}`.")
        data_type = TrainingDataType(name=data_type_name)
        data_type.save()
    # Save model file.
    path = f"static/{name}--{file.name}"  # Guaranteed unique.
    with open(BASE_DIR / path, "wb") as fd:
        fd.write(file.file.read())
    # Save model.
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
    server = scheduler.task
    if server is None:
        logger.error("No server running but got params to store.")
        return Response("No server running.", HTTP_400_BAD_REQUEST)
    file = file_in_request(request)
    if file is None:
        return Response("No file in request.", HTTP_400_BAD_REQUEST)
    params = file.file.read()
    to_save = ModelParams(params=params, tflite_model=server.model)
    to_save.save()
    server.update_session_end_time()
    return Response("ok")
