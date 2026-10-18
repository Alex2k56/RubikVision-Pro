from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/scan-face/", views.scan_face, name="scan_face"),
    path("api/solve/", views.solve, name="solve"),
]