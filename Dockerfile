FROM continuumio/miniconda3
WORKDIR /src
RUN apt-get update -q  \
    && apt-get install --no-install-recommends -qy g++ gcc inetutils-ping coinor-cbc\
    && rm -rf /var/lib/apt/lists/*
RUN conda create -n py39 python=3.9 pip
RUN echo "source activate py39" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
COPY requirements.txt .
COPY no_deps_requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-deps --no-cache-dir -r no_deps_requirements.txt
RUN conda config --set channel_priority false
COPY . .
