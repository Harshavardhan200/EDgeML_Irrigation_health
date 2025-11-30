import os

from mlops.config import PLANT_MODEL_DIR
from mlops.utils import create_version_dir, version_models
from src.plant_health import PlantHealthModel


def train_plant_health():
    """Train plant-health model, save a versioned snapshot, and return (acc, version_dir)."""
    print("🌿 Training PLANT HEALTH model...")

    model = PlantHealthModel()
    acc = model.train()

    if acc is None:
        print("⚠ Plant health training returned None (possibly empty dataset).")
        return 0.0, None

    print(f"🌿 Plant health accuracy: {acc:.4f}")

    # Save a version folder with timestamp + accuracy
    version_dir = create_version_dir(PLANT_MODEL_DIR, acc)
    version_models(PLANT_MODEL_DIR, version_dir)

    return acc, version_dir
