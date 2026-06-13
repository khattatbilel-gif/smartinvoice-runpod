import sys
print("HELLO FROM PYTHON", flush=True)

import torch
print(f"torch OK: {torch.__version__}", flush=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print("transformers imported OK", flush=True)

import runpod
print("runpod imported OK", flush=True)

model = None
processor = None

def load_model():
    global model, processor
    print("Loading model...", flush=True)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            max_pixels=1280 * 28 * 28,
        )
        print("Model ready!", flush=True)
    except Exception as e:
        print(f"Model loading FAILED: {e}", flush=True)

load_model()

def handler(job):
    return {"status": "ok", "model_loaded": model is not None}

runpod.serverless.start({"handler": handler})
