from django.urls import path

from .views import DigitDataset, DigitRecogApiView

urlpatterns = [
    path("recognize/", DigitRecogApiView.as_view()),
    path("dataset/", DigitDataset.as_view()),
]
