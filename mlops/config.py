from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_IRRIGATION = ROOT / "data" / "irrigation.csv"
DATA_PLANT = ROOT / "data" / "plant_health_data.csv"

MODELS_ROOT = ROOT / "models"

IRR_MODEL = MODELS_ROOT / "irrigation_model.pkl"
IRR_SCALER = MODELS_ROOT / "irrigation_scaler.pkl"
IRR_ENCODERS = MODELS_ROOT / "irrigation_encoders.pkl"

PLANT_MODEL = MODELS_ROOT / "plant_health_svm.pkl"
PLANT_SCALER = MODELS_ROOT / "plant_health_scaler.pkl"
PLANT_ENCODER = MODELS_ROOT / "plant_health_encoder.pkl"
