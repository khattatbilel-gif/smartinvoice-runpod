import sys
print("HELLO FROM PYTHON", flush=True)

import torch
print(f"torch OK: {torch.__version__}", flush=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print("transformers OK", flush=True)

print("Loading model...", flush=True)
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

import runpod

def handler(job):
    return {"status": "ok", "message": "model loaded successfully"}

runpod.serverless.start({"handler": handler})
