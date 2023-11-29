import os
from celery import Celery

CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

worker = Celery(
    "worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["fastapi_app.main"],
)

worker.conf.task_queues = {
    'default_queue': {
        'exchange': 'default_exchange',
        'routing_key': 'default',
    },
    'milp_queue': {
        'exchange': 'milp_exchange',
        'routing_key': 'milp',
    },

}

worker.conf.task_routes = {
    'celery_worker.task_grid_opt': {'queue': 'default_queue'},
    'celery_worker.task_supply_opt': {'queue': 'milp_queue'},
    'celery_worker.task_remove_anonymous_users': {'queue':'default_queue'},
}
