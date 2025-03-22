from django.urls import path

from .views import DigitDataset, DigitRecogApiView

urlpatterns = [
    path("", DigitRecogApiView.as_view()),
    path("dataset-info/", DigitDataset.as_view()),
]
