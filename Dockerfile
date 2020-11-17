FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      curl \
      python3-pip \
      python3-dev \
      python3-setuptools \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*
COPY requirement.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirement.txt