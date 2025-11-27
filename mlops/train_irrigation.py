import joblib
import pandas as pd
import os

from src.Irrigation_Model import IrrigationModel
from .config import DATA_IRRIGATION, IRR_MODEL, IRR_SCALER, IRR_ENCODERS

def main():
    model = IrrigationModel(
        dataset=str(DATA_IRRIGATION),
        model_file=str(IRR_MODEL)
    )
    
    model.train()

    print("[OK] Irrigation model retrained & overwritten in /models")

if __name__ == "__main__":
    main()
