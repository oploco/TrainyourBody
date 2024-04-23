from django.urls import path
from . import views

app_name = "cuerpoFit"
urlpatterns = [
    path('', views.predict_view, name='predict_view'),
]