FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /app

# Pin CUDA torch BEFORE qwen-vl-utils can overwrite it
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining deps (qwen-vl-utils won't overwrite torch now)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps qwen-vl-utils
RUN pip install --no-cache-dir qwen-vl-utils --no-deps

COPY handler.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "handler.py"]
