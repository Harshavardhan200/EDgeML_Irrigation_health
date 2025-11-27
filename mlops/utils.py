import joblib
from .config import *

def load_irrigation():
    model = joblib.load(IRR_MODEL)
    scaler = joblib.load(IRR_SCALER)
    encoders = joblib.load(IRR_ENCODERS)
    return model, scaler, encoders

def load_plant_health():
    model = joblib.load(PLANT_MODEL)
    scaler = joblib.load(PLANT_SCALER)
    encoder = joblib.load(PLANT_ENCODER)
    return model, scaler, encoder
