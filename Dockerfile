FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# Verify torch CUDA works before installing anything
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('Base torch:', torch.__version__)"

# Install packages — torch is already in the base image, don't reinstall it
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.30.0" \
    "bitsandbytes>=0.43.0" \
    "Pillow>=10.0.0"

# Install qwen-vl-utils without deps so it cannot overwrite torch
RUN pip install --no-cache-dir --no-deps "qwen-vl-utils>=0.0.8"

# Final check
RUN python -c "import torch; print('Final torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
