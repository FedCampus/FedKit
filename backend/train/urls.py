from django.urls import path
from train.views import *

urlpatterns = [
    path("advertised", advertise_model),
    path("server", request_server),
    path("upload", upload_file),
    path("params", store_params),
]
