import asyncio

from fastapi_app.python.db import async_inserts, sync_inserts, sync_queries
from fastapi_app.python.opt_models.grid_optimizer import optimize_grid
from fastapi_app.python.opt_models.supply_optimizer import optimize_energy_system
from fastapi_app.python.task_queue.celery_worker import worker


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


@worker.task(name='celery_worker.task_startup', force=True, track_started=True)
def task_startup():
    if not sync_queries.check_if_weather_data_exists():
        sync_inserts.update_weather_db()
    sync_inserts.create_default_user_account()

