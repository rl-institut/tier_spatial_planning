#!bin/bash
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
export C_FORCE_ROOT="true"
export BROKER_URL="redis://localhost:6379/0"
celery -A tasks worker --loglevel=info