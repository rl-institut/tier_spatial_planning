FROM python:3.6-alpine

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 5001
ENV DEBUG true

COPY . /flask_app
WORKDIR /flask_app

# install requirements
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install gunicorn

# expose the app port
EXPOSE 5001

# run the app server, the last argument match the app variable in the webapp.py file
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "3", "webapp:app"]