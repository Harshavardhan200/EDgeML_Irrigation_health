from mlops.train_irrigation import train_irrigation
from mlops.train_plant_health import train_plant_health
from mlops.metrics import load_last_metrics, save_metrics, should_rollback
from mlops.utils import rollback_irrigation, rollback_plant

def retrain_all():
    print("\n=== NIGHTLY RETRAIN START ===")

    last = load_last_metrics()

    # Train models
    irr_acc = train_irrigation()
    plant_acc = train_plant_health()

    print(f"Previous Irrigation Acc: {last['irrigation_acc']}, New: {irr_acc}")
    print(f"Previous Plant Acc: {last['plant_acc']}, New: {plant_acc}")

    rolled_back = False

    # Irrigation rollback
    if should_rollback(last["irrigation_acc"], irr_acc):
        rollback_irrigation()
        rolled_back = True

    # Plant Health rollback
    if should_rollback(last["plant_acc"], plant_acc):
        rollback_plant()
        rolled_back = True

    # Save new metrics ONLY IF NOT rolled back
    if not rolled_back:
        save_metrics(irr_acc, plant_acc)
        print("✔ Metrics updated with improved accuracies.")
    else:
        print("⚠️ Rollback applied. Metrics not updated.")

    print("=== NIGHTLY RETRAIN COMPLETE ===")
