FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 5001
ENV DEBUG true

COPY . /fastapi_app
WORKDIR /

# install requirements
RUN python -m pip install --upgrade pip
RUN pip install -r fastapi_app/requirements.txt
RUN pip install gunicorn
RUN ls

# expose the app port
EXPOSE 5001

# run the app server, the last argument match the app variable in the webapp.py file
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "5001"]