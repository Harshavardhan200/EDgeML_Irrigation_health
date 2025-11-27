import json
import os

METRICS_FILE = "mlops/last_metrics.json"

def load_last_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"irrigation_acc": 0, "plant_acc": 0}
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

def save_metrics(irrigation_acc, plant_acc):
    data = {
        "irrigation_acc": irrigation_acc,
        "plant_acc": plant_acc
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def should_rollback(old_acc, new_acc, min_improve=0.01):
    """
    If accuracy drops by more than 1% → rollback
    """
    return new_acc < old_acc - min_improve
