import os
import shutil
import joblib
from datetime import datetime
import subprocess

# ============================================================
# VERSIONING FUNCTIONS (NEEDED BY training scripts)
# ============================================================

def create_version_dir(base_dir, acc):
    from mlops.config import timestamp
    folder = f"{timestamp()}_acc_{acc:.4f}"
    version_dir = os.path.join(base_dir, "versions", folder)
    os.makedirs(version_dir, exist_ok=True)
    return version_dir



def save_current_model(src_path, dst_path):
    """Copy model files into current/."""
    os.makedirs(dst_path, exist_ok=True)
    for f in os.listdir(src_path):
        shutil.copy(os.path.join(src_path, f), os.path.join(dst_path, f))


def version_models(model_files, version_path):
    """Save multiple model files into versioned directory."""
    os.makedirs(version_path, exist_ok=True)
    for f in model_files:
        shutil.copy(f, os.path.join(version_path, os.path.basename(f)))


def git_commit_and_push(msg="Model Update"):
    """Commit and push to GitHub safely."""
    try:
        subprocess.run(["git", "add", "."], check=False)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("⚠️ Git push failed:", e)

# ============================================================
# ROLLBACK FUNCTIONS
# ============================================================

def rollback_irrigation():
    versions_path = "models/irrigation/versions/"
    versions = sorted(os.listdir(versions_path))
    if len(versions) >= 2:
        prev = versions[-2]  # previous version
        src = os.path.join(versions_path, prev)
        dst = "models/irrigation/current/"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✔ Irrigation rolled back to {prev}")


def rollback_plant():
    versions_path = "models/plant_health/versions/"
    versions = sorted(os.listdir(versions_path))
    if len(versions) >= 2:
        prev = versions[-2]
        src = os.path.join(versions_path, prev)
        dst = "models/plant_health/current/"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✔ Plant Health rolled back to {prev}")
        
