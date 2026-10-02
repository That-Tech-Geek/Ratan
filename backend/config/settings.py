import os
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
SECRET_KEY=os.getenv("DJANGO_SECRET_KEY","dev-only-change-me")
DEBUG=os.getenv("DJANGO_DEBUG","false").lower()=="true"
ALLOWED_HOSTS=os.getenv("DJANGO_ALLOWED_HOSTS","*").split(",")
INSTALLED_APPS=["django.contrib.auth","django.contrib.contenttypes","django.contrib.sessions","django.contrib.messages","django.contrib.staticfiles","rest_framework","corsheaders","gyaan"]
MIDDLEWARE=["corsheaders.middleware.CorsMiddleware","django.middleware.security.SecurityMiddleware","django.contrib.sessions.middleware.SessionMiddleware","django.middleware.common.CommonMiddleware","django.middleware.csrf.CsrfViewMiddleware","django.contrib.auth.middleware.AuthenticationMiddleware","django.contrib.messages.middleware.MessageMiddleware"]
ROOT_URLCONF="config.urls";WSGI_APPLICATION="config.wsgi.application";TEMPLATES=[]
DATABASES={"default":{"ENGINE":"django.db.backends.postgresql","NAME":os.getenv("POSTGRES_DB","gyaan"),"USER":os.getenv("POSTGRES_USER","gyaan"),"PASSWORD":os.getenv("POSTGRES_PASSWORD","gyaan"),"HOST":os.getenv("POSTGRES_HOST","localhost"),"PORT":os.getenv("POSTGRES_PORT","5432")}}
if os.getenv("CI")=="true":DATABASES={"default":{"ENGINE":"django.db.backends.sqlite3","NAME":BASE_DIR/"ci.sqlite3"}}
CORS_ALLOW_ALL_ORIGINS=True;DEFAULT_AUTO_FIELD="django.db.models.BigAutoField";STATIC_URL="/static/"