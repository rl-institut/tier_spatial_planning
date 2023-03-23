import os
from celery import Celery

CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

worker = Celery("worker", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND, include=["fastapi_app.main"])
worker.conf.update({'CELERY_TASK_TRACK_STARTED': True,
                    'CELERY_ACCEPT_CONTENT': ['application/json'],
                    'CELERY_RESULT_SERIALIZER': 'json',
                    'CELERY_TASK_SERIALIZER': 'json',
                    'CELERY_IGNORE_RESULT': True,
                    'CELERY_TASK_IGNORE_RESULT': True,
                    'CELERY_TRACK_STARTED': True})
