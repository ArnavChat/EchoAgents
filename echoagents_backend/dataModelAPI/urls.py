from django.urls import path
from .views import reddit_opinion_view

urlpatterns = [
    path('api/reddit-opinion/', reddit_opinion_view),
]
