from flask import Flask, url_for, render_template, redirect, jsonify

try:
    from worker import celery
except ModuleNotFoundError:
    from .worker import celery
import celery.states as states

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_simulation")
def run_simulation():
    input_json = simulation_output = {
        "name": "dummy_json_input",
        "secondary_dict": {"val1": 2, "val2": [5, 6, 7, 8]},
    }
    task = celery.send_task("tasks.run_simulation", args=[input_json], kwargs={})

    return render_template("submitted_task.html", task_id=task.id)


@app.route("/check/<string:task_id>")
def check_task(task_id: str) -> str:
    res = celery.AsyncResult(task_id)
    if res.state == states.PENDING:
        return res.state
    else:
        return jsonify(res.result)
