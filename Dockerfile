FROM python:3.10-slim

WORKDIR /app

# Install CUDA torch FIRST before anything else
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install all other packages (except qwen-vl-utils)
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    Pillow>=10.0.0

# Install qwen-vl-utils WITHOUT dependencies so it cannot overwrite torch
RUN pip install --no-cache-dir --no-deps qwen-vl-utils>=0.0.8

COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
