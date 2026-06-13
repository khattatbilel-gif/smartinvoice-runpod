import sys
print("HELLO FROM PYTHON", flush=True)

import torch
print(f"torch OK: {torch.__version__}", flush=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print("transformers OK", flush=True)

import runpod

def handler(job):
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available()
    }

runpod.serverless.start({"handler": handler})
