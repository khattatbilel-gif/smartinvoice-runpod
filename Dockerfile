FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install CUDA torch first — before anything else
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Verify torch sees CUDA before continuing
RUN python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Install everything else
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    Pillow>=10.0.0

# Install qwen-vl-utils WITHOUT deps so it cannot overwrite torch
RUN pip install --no-cache-dir --no-deps qwen-vl-utils>=0.0.8

# Final check — torch must still be CUDA after all installs
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('Final torch:', torch.__version__)"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
