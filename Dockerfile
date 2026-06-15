FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Pin numpy FIRST to prevent 2.x from being pulled in
RUN pip install --no-cache-dir "numpy<2.0"

WORKDIR /app

# Install CUDA torch with pinned versions
RUN pip install --no-cache-dir \
    "torch==2.2.0+cu121" \
    "torchvision==0.17.0+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Install other deps with pinned torch to prevent overwrite
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.30.0" \
    "bitsandbytes>=0.43.0" \
    "Pillow>=10.0.0" \
    "torch==2.2.0+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --no-deps "qwen-vl-utils>=0.0.8"

RUN python -c "import torch; print('Final torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
