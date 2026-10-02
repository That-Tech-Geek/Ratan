from django.urls import path
from gyaan.api import health,questions,sync_batch
urlpatterns=[path("api/v1/health/",health),path("api/v1/diagnostic/questions",questions),path("api/v1/sync/batch/",sync_batch)]