import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from Irrigation_Model import IrrigationModel
from plant_health import PlantHealthModel


# ---------------------------------------------------
# CONFIGURE LOGGING
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------
# 1. INITIALIZE BOTH ML MODELS
# ---------------------------------------------------
logging.info("Initializing Irrigation and Plant Health Models...")

irrigation_model = IrrigationModel("irrigation.csv")
irrigation_model.train()

plant_model = PlantHealthModel("plant_health_data.csv")
plant_model.train()

logging.info("Models Loaded Successfully.")


# ---------------------------------------------------
# 2. GET SENSOR READINGS (replace with real RPi readings)
# ---------------------------------------------------
soil_type = "Black Soil"
stage = "Germination"
moi_raw = 600            # ADC raw value
temp = 26.2
humidity = 61
light = 480
nitrogen = 18
phosphorus = 30
potassium = 40

logging.info("Sensor values loaded.")


# ---------------------------------------------------
# 3. RUN ML PREDICTIONS
# ---------------------------------------------------
logging.info("Running predictions...")

irrigation_pred, moisture_percent = irrigation_model.predict(
    soil_type=soil_type,
    stage=stage,
    moi_raw=moi_raw,
    temp=temp,
    humidity=humidity
)

plant_pred = plant_model.predict(
    soil_moisture=moisture_percent,
    temp=temp,
    humidity=humidity,
    light=light,
    nitrogen=nitrogen,
    phosphorus=phosphorus,
    potassium=potassium
)

logging.info(f"Irrigation Prediction: {irrigation_pred}")
logging.info(f"Plant Health Prediction: {plant_pred}")


# ---------------------------------------------------
# 4. LOAD LOCAL HUGGINGFACE LLM
# ---------------------------------------------------
logging.info("Loading Hugging Face flan-t5-small model...")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
hf_model = hf_model.to("cpu")

logging.info("LLM Ready.")


# ---------------------------------------------------
# 5. BUILD PROMPT (IRRIGATION + PLANT HEALTH)
# ---------------------------------------------------
prompt = f"""
You are an agricultural expert AI system.

Here are live sensor readings:
- Soil Type: {soil_type}
- Growth Stage: {stage}
- Soil Moisture (%): {moisture_percent}
- Temperature (°C): {temp}
- Humidity (%): {humidity}
- Light Intensity (lux): {light}
- Nitrogen Level: {nitrogen}
- Phosphorus Level: {phosphorus}
- Potassium Level: {potassium}

Machine learning predictions:
- Irrigation Need (0 = No, 1 = Yes): {irrigation_pred}
- Plant Health Status: {plant_pred}

Now generate a detailed guidance report including:
1. Explanation of irrigation requirement.
2. Explanation of plant health condition.
3. Corrective actions.
4. Soil moisture management.
5. Fertilizer and nutrient corrections based on NPK.
6. Light/shade recommendations.
7. Watering quantity and frequency.
8. Next 3–5 day maintenance plan.
9. Warning signs to monitor.
10. Steps for preventing stress.

Write the answer in clear bullet points.
"""


logging.info("Generating LLM response...")


# ---------------------------------------------------
# 6. GENERATE A LONG RESPONSE
# ---------------------------------------------------

inputs = tokenizer(prompt, return_tensors="pt")
outputs = hf_model.generate(**inputs, max_length=200)
print(outputs)

advice = tokenizer.decode(outputs[0], skip_special_tokens=True)

logging.info("LLM Response Generated Successfully.")


# ---------------------------------------------------
# 7. LOG THE FINAL AGRICULTURAL ADVICE
# ---------------------------------------------------
logging.info("---- AI AGRICULTURAL RECOMMENDATION REPORT ----")
logging.info(advice)
logging.info("------------------------------------------------")
