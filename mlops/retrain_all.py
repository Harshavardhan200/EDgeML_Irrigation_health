from .train_irrigation import main as train_irrigation
from .train_plant_health import main as train_plant

def main():
    print("=== Retraining Irrigation Model ===")
    train_irrigation()

    print("=== Retraining Plant Health Model ===")
    train_plant()

    print("=== ALL MODELS RETRAINED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
