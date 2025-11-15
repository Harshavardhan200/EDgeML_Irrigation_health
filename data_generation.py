import time
import json
import ssl
import logging
import random
import paho.mqtt.client as mqtt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Irrigation_Model import IrrigationModel
from plant_health import PlantHealthModel

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================================================
# MQTT SETTINGS (HiveMQ Cloud)
# ======================================================
MQTT_BROKER = "8c70285096fe43429db68ea8e5513422.s1.eu.hivemq.cloud"
MQTT_PORT = 8883

USERNAME = "hivemq.webclient.1763167024884"
PASSWORD = "!A5PgmOd1MS$<7z9X#bf"

TOPIC_SENSOR = "agriedge/sensor"
TOPIC_ADVICE = "agriedge/advice"

# ======================================================
# SENSOR IMPORTS (UNCOMMENT ON RPI)
# ======================================================
# import adafruit_dht
# from gpiozero import MCP3008

# dht = adafruit_dht.DHT11(4)
# soil_adc = MCP3008(channel=0)
# ldr_adc = MCP3008(channel=1)

# ======================================================
# MQTT CLIENT SETUP
# ======================================================
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)

client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)

logging.info("Connecting to HiveMQ Cloud...")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

logging.info("MQTT Connected and Publishing Enabled.")

# ======================================================
# LOAD ML MODELS
# ======================================================
logging.info("Loading ML Models...")

irrigation_model = IrrigationModel("irrigation.csv")
irrigation_model.train()

plant_model = PlantHealthModel("plant_health_data.csv")
plant_model.train()

logging.info("Irrigation & Plant Health Models Loaded.")

# ======================================================
# LOAD LLM MODEL (HUGGINGFACE)
# ======================================================
logging.info("Loading Hugging Face flan-t5-small...")

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")

logging.info("LLM Loaded Successfully.")

# ======================================================
# MAIN LOOP
# ======================================================
logging.info("System Running... Reading sensors + ML + LLM + MQTT")

while True:
    try:
        # ----------------------------------------
        # REAL SENSOR CODE  (ENABLE ON RPI)
        # ----------------------------------------
        # temperature = dht.temperature
        # humidity = dht.humidity
        #
        # soil_raw = soil_adc.value * 1023
        # soil_moisture = round((soil_raw / 1023) * 100, 2)
        #
        # ldr_raw = ldr_adc.value * 1023
        # light = int(ldr_raw)

        # ----------------------------------------
        # SIMULATED VALUES (USE ON LAPTOP)
        # ----------------------------------------
        temperature = round(random.uniform(22, 29), 2)
        humidity = round(random.uniform(40, 70), 2)
        soil_moisture = round(random.uniform(10, 75), 2)
        light = random.randint(200, 700)
        nitrogen = random.randint(10, 30)
        phosphorus = random.randint(10, 30)
        potassium = random.randint(10, 30)

        soil_type = "Black Soil"
        stage = "Germination"

        # ---------------------------------------------------
        # ML PREDICTION
        # ---------------------------------------------------
        irrigation_pred, moisture_percent = irrigation_model.predict(
            soil_type=soil_type,
            stage=stage,
            moi_raw=soil_moisture,
            temp=temperature,
            humidity=humidity
        )

        plant_pred = plant_model.predict(
            soil_moisture=moisture_percent,
            temp=temperature,
            humidity=humidity,
            light=light,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium
        )

        logging.info(f"Irrigation Need: {irrigation_pred}")
        logging.info(f"Plant Health: {plant_pred}")

        # ---------------------------------------------------
        # BUILD LLM PROMPT
        # ---------------------------------------------------
        prompt = f"""
You are an agricultural expert AI system.

Sensor Inputs:
- Soil Type: {soil_type}
- Growth Stage: {stage}
- Soil Moisture: {moisture_percent}
- Temperature: {temperature}
- Humidity: {humidity}
- Light: {light}
- Nitrogen: {nitrogen}
- Phosphorus: {phosphorus}
- Potassium: {potassium}

ML Predictions:
- Irrigation Needed (0/1): {irrigation_pred}
- Plant Health: {plant_pred}

Generate detailed guidance:
1. Is irrigation required? Why?
2. Explain plant health condition.
3. Recommended next actions.
4. Fertilizer/NPK corrections.
5. Light/shade correction.
6. Watering schedule.
7. 3–5 day maintenance plan.
8. Warning signs to monitor.
9. Prevention guidelines.
"""

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = hf_model.generate(**inputs, max_length=250)
        advice = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info("LLM Advice Generated.")

        # ---------------------------------------------------
        # MQTT PUBLISH SENSOR + PREDICTIONS
        # ---------------------------------------------------
        payload = {
            "temperature": temperature,
            "humidity": humidity,
            "moisture": moisture_percent,
            "light": light,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "irrigation_prediction": irrigation_pred,
            "plant_health_prediction": plant_pred,
            "timestamp": time.time()
        }

        client.publish(TOPIC_SENSOR, json.dumps(payload))
        client.publish(TOPIC_ADVICE, advice)

        logging.info("MQTT Published Sensor Data + Advice")
        logging.info("---------------------------")
        logging.info(advice)
        logging.info("---------------------------")

    except Exception as e:
        logging.error(f"Error: {e}")

    time.sleep(10)
