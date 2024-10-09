import os

from celery import Celery

"""
This module sets up a Celery worker for asynchronous task processing in the FastAPI application. It configures the 
Celery worker with a broker and result backend, both typically using Redis. The worker includes tasks defined in the 
"fastapi_app.main" module.

The configuration defines two task queues: default_queue and milp_queue, with their respective exchanges and routing 
keys. This setup allows for categorizing tasks based on their nature or processing requirements.

Specific tasks are routed to these queues: task_grid_opt and task_remove_anonymous_users are directed to the 
default_queue, while task_supply_opt is routed to the milp_queue. This segregation helps in managing task execution 
based on priority or resource requirements, making the system efficient and scalable.
"""

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
    'celery_worker.task_remove_anonymous_users': {'queue': 'default_queue'},
}
