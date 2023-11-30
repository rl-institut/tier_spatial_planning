FROM continuumio/miniconda3
WORKDIR /src
RUN apt-get update -q  \
    && apt-get install --no-install-recommends -qy g++ gcc inetutils-ping coinor-cbc \
    && rm -rf /var/lib/apt/lists/*
RUN conda update -n base -c defaults conda
RUN conda create -n py38 python=3.8 pip
RUN echo "source activate py38" > ~/.bashrc
ENV PATH /opt/conda/envs/py38/bin:$PATH
COPY requirements.txt .
COPY no_deps_requirements.txt .
RUN pip install --upgrade pip || true
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-deps --no-cache-dir -r no_deps_requirements.txt
RUN conda config --set channel_priority false
COPY .. .
