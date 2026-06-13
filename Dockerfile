FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /app

# Install everything EXCEPT torch and qwen-vl-utils first
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    Pillow>=10.0.0

# Install qwen-vl-utils WITHOUT its dependencies (prevents torch overwrite)
RUN pip install --no-cache-dir --no-deps qwen-vl-utils>=0.0.8

COPY handler.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "handler.py"]
