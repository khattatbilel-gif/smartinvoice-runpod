# SmartInvoice — RunPod Serverless Docker Image
# Base: RunPod's official PyTorch image with CUDA 11.8
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Pre-download the model weights at build time so cold starts are faster.
# Comment this out if you prefer to pull weights at runtime (smaller image, slower cold start).
RUN python -c "\
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; \
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', max_pixels=1280*28*28); \
print('Processor downloaded.')"

# Note: full model weights (~15 GB) are large to bake into the image.
# Alternative: mount a RunPod network volume with pre-cached weights and
# set HF_HOME=/runpod-volume/hf_cache in your endpoint environment variables.
# This makes cold starts ~5s instead of ~90s.

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]