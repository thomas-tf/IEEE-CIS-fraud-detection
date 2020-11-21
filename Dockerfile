FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
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
    pip3 install -r requirement.txt && \
    export CUDA=cu110 TORCH=1.7.0 && \
    pip install torch==${TORCH}+${CUDA} torchvision==0.8.1+${CUDA} torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && \
    pip install torch-geometric
