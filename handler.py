import os, re, json, traceback, base64, io
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import runpod

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Loading model...")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LOAD_ERROR = None
model = None
processor = None

try:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL,
        max_pixels=1280 * 28 * 28,
        cache_dir=os.environ.get("HF_HOME", None)
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None)
    )
    print("Model loaded successfully.")
except Exception:
    LOAD_ERROR = traceback.format_exc()
    print("Model load FAILED:\n", LOAD_ERROR)

# ... rest of handler stays exactly the same ...
