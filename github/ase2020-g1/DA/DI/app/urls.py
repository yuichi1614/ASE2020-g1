from django.conf.urls import url
from django.views.static import serve

from DI.settings import MEDIAS_ROOT
from app.views import allPage, refresh

urlpatterns = [
    url(r'^$', allPage),
    url(r'^refresh/$', refresh),
    url(r'^medias/(?P<path>.*)$', serve, {'document_root': MEDIAS_ROOT}),
]
