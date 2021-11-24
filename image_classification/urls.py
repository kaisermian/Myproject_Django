from django.urls import path
from image_classification import views

urlpatterns = [
    path('', views.DetectionPage.as_view(), name="home_page")
]