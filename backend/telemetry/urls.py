from django.urls import path
from telemetry.views import *

urlpatterns = [path("evaluate_ins", evaluate_ins), path("fit_ins", fit_ins)]
