FROM continuumio/anaconda3
WORKDIR /src
RUN apt update
RUN conda create -n py39 python=3.9 pip
RUN echo "source activate py10" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
COPY fastapi_app/requirements.txt .
COPY ./fastapi_app/requirements.txt ./fastapi_app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN conda config --set channel_priority false
RUN conda install -c conda-forge coincbc
COPY ./fastapi_app ./fastapi_app
