FROM continuumio/anaconda3
WORKDIR /src
RUN apt-get update -q  \
    && apt-get install --no-install-recommends -qy g++ gcc inetutils-ping  \
    && rm -rf /var/lib/apt/lists/*
RUN conda create -n py39 python=3.9 pip
RUN echo "source activate py10" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
COPY fastapi_app/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN conda config --set channel_priority false
RUN conda install -c conda-forge coincbc
COPY ./fastapi_app ./fastapi_app
