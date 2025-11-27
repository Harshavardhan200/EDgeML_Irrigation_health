

from src.plant_health import PlantHealthModel

from .config import DATA_PLANT, PLANT_MODEL

def main():
    model = PlantHealthModel(
        dataset=str(DATA_PLANT),
        model_file=str(PLANT_MODEL)
    )

    model.train()

    print("[OK] Plant Health SVM retrained & overwritten in /models")

if __name__ == "__main__":
    main()
