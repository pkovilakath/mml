from django.conf.urls import url

from .views import learn
from .views import saveModel
from .views import predict


urlpatterns = [
    url(r'home$', learn),
    url(r'learn', learn),
    url(r'predict', predict),
    url(r'saveModel$', saveModel)
]