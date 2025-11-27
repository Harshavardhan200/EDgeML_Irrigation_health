import os
import shutil

def rollback_irrigation():
    versions_path = "models/irrigation/versions/"
    versions = sorted(os.listdir(versions_path))
    if len(versions) >= 2:
        prev_version = versions[-2]
        src = os.path.join(versions_path, prev_version)
        dst = "models/irrigation/current/"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✔ Irrigation model rolled back to {prev_version}")


def rollback_plant():
    versions_path = "models/plant_health/versions/"
    versions = sorted(os.listdir(versions_path))
    if len(versions) >= 2:
        prev_version = versions[-2]
        src = os.path.join(versions_path, prev_version)
        dst = "models/plant_health/current/"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✔ Plant Health model rolled back to {prev_version}")
