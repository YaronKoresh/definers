import runpod
from definers import start

def handler(event):
    proj = event["project"]
    return start(proj)

runpod.serverless.start({"handler": handler})