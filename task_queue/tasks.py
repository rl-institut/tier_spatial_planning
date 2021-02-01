import os
import time
import json
from celery import Celery


CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@celery.task(name="tasks.run_simulation")
def run_simulation(simulation_input: dict,) -> dict:
    time.sleep(5)
    # TODO run simulation with `simulation_input`
    simulation_output = {
        "name": "dummy_json_return",
        "secondary_dict": {"val1": 1, "val2": [1, 2, 3, 4]},
    }
    return json.dumps(simulation_output)
