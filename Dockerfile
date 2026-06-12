FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Model weights are downloaded at runtime from HuggingFace.
# Set HF_HOME env variable in RunPod to cache them on a network volume.

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
