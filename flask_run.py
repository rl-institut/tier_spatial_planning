"""
From the root of the repository, type `python flask_run.py` to start the flask server
"""
from flask_app.webapp import app

if __name__ == "__main__":
    app.run(debug=True, port=5001)
