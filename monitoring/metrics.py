import json
from pathlib import Path
from datetime import datetime, timezone

LOG_FILE = Path("monitoring/api_metrics.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)


def log_request(data: dict):
    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
