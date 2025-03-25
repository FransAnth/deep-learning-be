from django.urls import path

from .views import TicTacToe

urlpatterns = [path("", TicTacToe.as_view())]
