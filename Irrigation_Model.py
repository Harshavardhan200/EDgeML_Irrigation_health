import os
import csv
import pandas as pd
import adafruit_dht
from gpiozero import MCP3008
import time
import joblib
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# --------------------------
# CONFIGURE LOGGING
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class IrrigationModel:
    def __init__(self, dataset="irrigation.csv", model_file="irrigation_model.pkl"):
        self.dataset = dataset
        self.model_file = model_file
        self.model = None
        self.scaler = StandardScaler()

        # Label encoders for categorical columns
        self.encoders = {
            "soil_type": LabelEncoder(),
            "Seedling Stage": LabelEncoder()
        }

        logging.info("IrrigationModel initialized.")

    # -----------------------------------------
    def load_dataset(self):
        """Load dataset fresh every time."""
        df = pd.read_csv(self.dataset)

        # Clean unnecessary columns
        for col in ["Unnamed: 0", "crop_ID"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        logging.info(f"Dataset loaded with shape {df.shape}.")
        return df

    # -----------------------------------------
    def preprocess(self, df):
        """Encode + scale features."""
        df = df.copy()

        # Encode categorical columns
        for col in self.encoders:
            df[col] = self.encoders[col].fit_transform(df[col])

        X = df[["soil_type", "Seedling Stage", "MOI", "temp", "humidity"]]
        y = df["result"]

        # Scale numeric features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    # -----------------------------------------
    def train(self):
        """Train the SVM model."""
        df = self.load_dataset()

        # Only rows with result 0/1
        df = df[df["result"].isin([0, 1])]

        if df.empty:
            logging.warning("Training skipped – no labeled data.")
            return

        X, y = self.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = SVC(kernel="rbf", probability=True)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        logging.info(f"Training complete. Accuracy = {acc}")
        logging.info("\n" + classification_report(y_test, preds))

        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, "irrigation_scaler.pkl")
        joblib.dump(self.encoders, "irrigation_encoders.pkl")

        return acc

    # -----------------------------------------
    def predict(self, soil_type, stage, moi_raw, temp, humidity):
        """Predict irrigation need using input values."""
        # Convert MOI from raw ADC value (0–1023) to % moisture
        moi = round((moi_raw / 1023) * 100, 2)

        if self.model is None:
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load("irrigation_scaler.pkl")
            self.encoders = joblib.load("irrigation_encoders.pkl")

        data = pd.DataFrame([{
            "soil_type": soil_type,
            "Seedling Stage": stage,
            "MOI": moi,
            "temp": temp,
            "humidity": humidity
        }])

        # Encode categoricals
        for col in self.encoders:
            data[col] = self.encoders[col].transform(data[col])

        X_scaled = self.scaler.transform(data)
        pred = self.model.predict(X_scaled)[0]

        logging.info(f"Prediction → {pred} for data {data.to_dict(orient='records')}")
        return pred, moi

    # -----------------------------------------
    def retrain(self):
        """Auto retrain using updated CSV."""
        logging.info("Retraining started...")
        return self.train()

