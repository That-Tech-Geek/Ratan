from django.urls import path
from gyaan.api import health,questions,likert_items,sync_batch,create_diagnostic_session
urlpatterns=[path("api/v1/health/",health),path("api/v1/diagnostic/questions",questions),path("api/v1/diagnostic/sessions",create_diagnostic_session),path("api/v1/likert/items",likert_items),path("api/v1/sync/batch/",sync_batch)]