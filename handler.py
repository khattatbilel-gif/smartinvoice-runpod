import sys
print("HELLO FROM PYTHON", flush=True)
print(f"Python version: {sys.version}", flush=True)

import runpod

def handler(job):
    return {"status": "ok", "message": "container works"}

runpod.serverless.start({"handler": handler})
