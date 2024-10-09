import asyncio

from fastapi_app.python.db import async_inserts, sync_inserts, sync_queries
from fastapi_app.python.opt_models.grid_optimizer import optimize_grid
from fastapi_app.python.opt_models.supply_optimizer import optimize_energy_system
from fastapi_app.python.task_queue.celery_worker import worker

"""
This module in a FastAPI application uses Celery to handle asynchronous tasks:

1. `task_grid_opt`: Optimizes grid layouts for users and projects, with retry capabilities.
2. `task_supply_opt`: Optimizes energy supply systems for specific users and projects.
3. `task_remove_anonymous_users`: Deletes anonymous user accounts asynchronously.

Additionally, it includes functions to check the status of these tasks, identifying if they have completed, failed, 
or been revoked. This setup enables efficient, asynchronous processing of complex tasks and user management.
"""

@worker.task(name='celery_worker.task_grid_opt',
             force=True,
             track_started=True,
             autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 1, 'countdown': 10})
def task_grid_opt(user_id, project_id):
    result = optimize_grid(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_supply_opt',
             force=True,
             track_started=True,
             autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 1, 'countdown': 10}
             )
def task_supply_opt(user_id, project_id):
    result = optimize_energy_system(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_remove_anonymous_users', force=True, track_started=True)
def task_remove_anonymous_users(user_id):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        result = loop.run_until_complete(async_inserts.remove_account(user_id))
    else:
        result = asyncio.run(async_inserts.remove_account(user_id))
    return result


def get_status_of_task(task_id):
    status = worker.AsyncResult(task_id).status.lower()
    return status


def task_is_finished(task_id):
    status = get_status_of_task(task_id)
    if status in ['success', 'failure', 'revoked']:
        return True
    else:
        return False

