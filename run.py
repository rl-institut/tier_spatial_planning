import uvicorn
from fastapi_app.main import app


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=40000)
    # go to http://localhost:127.0.0.1:40000/