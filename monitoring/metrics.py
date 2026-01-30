import json
import time
from pathlib import Path

log_file= Path("experiments/api_metrics.jsonl")

def log_request(data: dict):
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file,"a") as f:
        f.write(json.dump(data)+"\n")
        