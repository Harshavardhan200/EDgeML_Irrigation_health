import os
from mlops.config import DATA_PATH, IRRIGATION_MODEL_DIR
from mlops.utils import create_version_dir, version_models, git_commit_and_push
from src.Irrigation_Model import IrrigationModel

def train_irrigation():

    print("🌱 Training IRRIGATION model...")

    csv_path = os.path.join(DATA_PATH, "irrigation.csv")

    model = IrrigationModel()
    acc = model.train_from_csv(csv_path)

    # Save model to "current"
    model.save_all(IRRIGATION_MODEL_DIR)

    # Save version folder INCLUDING ACCURACY
    version_dir = create_version_dir(IRRIGATION_MODEL_DIR, acc)
    version_models(os.path.join(IRRIGATION_MODEL_DIR, "current"), version_dir)

    # Push
    git_commit_and_push(f"Updated irrigation model | acc={acc:.4f}")

    print("✔ IRRIGATION retraining complete.")
    return acc
