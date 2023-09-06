from django.urls import path
from train.views import (
    advertise_model,
    request_server,
    store_params,
    upload_model,
)

urlpatterns = [
    path("advertised", advertise_model),
    path("server", request_server),
    path("upload", upload_model),
    path("params", store_params),
]
