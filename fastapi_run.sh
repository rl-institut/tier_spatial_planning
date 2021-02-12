#!bin/bash
# From the root of the repository, type `python flask_run.py` to start the flask server
uvicorn fastapi_app.webapp:app --reload --port 5001
