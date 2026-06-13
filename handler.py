import sys
print("HELLO FROM PYTHON", flush=True)

import torch
print(f"torch OK: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

import runpod

def handler(job):
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available()
    }

runpod.serverless.start({"handler": handler})
