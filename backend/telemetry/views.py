import logging

from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import Request
from telemetry.serializers import *

logger = logging.getLogger(__name__)


def save_serializer_and_respond(serializer) -> Response:
    if serializer.is_valid():
        serializer.save()
    else:
        logger.error(serializer.errors)
    return Response("")


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def fit_ins(request: Request):
    serializer = FitInsTelemetryDataSerializer(data=request.data)  # type: ignore
    return save_serializer_and_respond(serializer)


@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def evaluate_ins(request: Request):
    serializer = EvaluateInsTelemetryDataSerializer(data=request.data)  # type: ignore
    return save_serializer_and_respond(serializer)
