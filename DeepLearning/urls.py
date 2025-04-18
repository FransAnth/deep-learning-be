from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("digit-recognition/", include("DigitRecognition.urls")),
    path("tic-tac-toe/", include("TicTacToe.urls")),
]
