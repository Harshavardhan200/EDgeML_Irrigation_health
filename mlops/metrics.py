"""
Metrics helpers for irrigation + plant health models.

We store the last *best* accuracies so that:
- `current/` always points to the best known model
- version folders can be compared against previous performance
"""
import json
import os
from mlops.config import PROJECT_ROOT, timestamp

# Store metrics at project_root/mlops/last_metrics.json
METRICS_FILE = os.path.join(PROJECT_ROOT, "mlops", "last_metrics.json")


def load_last_metrics():
    """Load previous best accuracies (or defaults if first run)."""
    if not os.path.exists(METRICS_FILE):
        return {
            "irrigation_acc": 0.0,
            "plant_acc": 0.0,
            "timestamp": None,
        }
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def save_metrics(irrigation_acc: float, plant_acc: float) -> None:
    """Persist the latest *best* accuracies to disk."""
    data = {
        "irrigation_acc": float(irrigation_acc),
        "plant_acc": float(plant_acc),
        "timestamp": timestamp(),
    }
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print("📊 Metrics file updated:", METRICS_FILE)


def should_rollback(old_acc: float, new_acc: float, min_improve: float = 0.0) -> bool:
    """Return True if new model is clearly worse than old one.

    By default (min_improve=0.0), this is simply: new_acc < old_acc.
    If you set min_improve > 0, then a small drop within the margin
    will *not* count as "worse enough" to roll back.
    """
    return new_acc < (old_acc - min_improve)
