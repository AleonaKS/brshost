# example/urls.py
from django.urls import path

from books.views import index


urlpatterns = [
    path('', index),
]