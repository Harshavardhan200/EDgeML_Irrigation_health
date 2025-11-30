"""
Utility helpers for model versioning, rollback, and git integration.
"""
import os
import shutil
import subprocess
from typing import Optional, List

from mlops.config import PROJECT_ROOT, timestamp


# =========================================
# VERSION FOLDER HELPERS
# =========================================
def create_version_dir(model_dir: str, acc: float) -> str:
    """Create a version folder like:
    models/<name>/versions/2025-11-30_23-59-59_acc_0.9523/
    and return its absolute path.
    """
    folder = f"{timestamp()}_acc_{acc:.4f}"
    version_dir = os.path.join(model_dir, "versions", folder)
    os.makedirs(version_dir, exist_ok=True)
    return version_dir


def version_models(model_dir: str, version_dir: str) -> None:
    """Copy all top-level .pkl files from model_dir into version_dir.

    This will copy, e.g.:
      models/irrigation/irrigation_model.pkl
      models/irrigation/irrigation_scaler.pkl
      models/irrigation/irrigation_encoders.pkl
    into:
      models/irrigation/versions/<timestamp_acc_x.xxx>/
    """
    os.makedirs(version_dir, exist_ok=True)

    for name in os.listdir(model_dir):
        src = os.path.join(model_dir, name)
        # Only copy top-level .pkl files (skip subfolders like current/ and versions/)
        if os.path.isfile(src) and name.endswith(".pkl"):
            dst = os.path.join(version_dir, name)
            shutil.copy2(src, dst)

    print(f"📦 Saved versioned models → {version_dir}")


def list_versions(model_dir: str) -> List[str]:
    """Return sorted list of version folder names (not full paths)."""
    versions_root = os.path.join(model_dir, "versions")
    if not os.path.exists(versions_root):
        return []
    return sorted(
        [d for d in os.listdir(versions_root) if os.path.isdir(os.path.join(versions_root, d))]
    )


def latest_version_dir(model_dir: str) -> Optional[str]:
    """Return absolute path to latest version folder, or None if none exist."""
    versions_root = os.path.join(model_dir, "versions")
    versions = list_versions(model_dir)
    if not versions:
        return None
    latest = versions[-1]
    return os.path.join(versions_root, latest)


def set_current_from_version_dir(model_dir: str, version_dir: str) -> None:
    """Copy all files from a given version_dir into model_dir/current."""
    current_dir = os.path.join(model_dir, "current")
    os.makedirs(current_dir, exist_ok=True)
    shutil.copytree(version_dir, current_dir, dirs_exist_ok=True)
    print(f"🔁 Updated current model for {model_dir} from {version_dir}")


def rollback_to_previous(model_dir: str) -> bool:
    """Set current/ to the previous (second-latest) version.

    Returns True if rollback happened, False otherwise.
    """
    versions_root = os.path.join(model_dir, "versions")
    versions = list_versions(model_dir)
    if len(versions) < 2:
        print(f"⚠ Not enough versions to rollback in {versions_root}")
        return False

    prev = versions[-2]
    src = os.path.join(versions_root, prev)
    dst = os.path.join(model_dir, "current")
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"🔄 Rolled back {model_dir} to version {prev}")
    return True


# =========================================
# OPTIONAL: SAFE GIT COMMIT AND PUSH (for local use)
# =========================================
def git_commit_and_push(message: str) -> None:
    """Commit and push changes using the local git configuration.

    In CI (CircleCI), it's usually better to commit/push from the pipeline
    config directly. This helper is mostly for manual/local workflows.
    """
    try:
        # Ensure we are in the project root
        os.chdir(PROJECT_ROOT)

        # Fetch / rebase to avoid simple conflicts
        subprocess.run(["git", "pull", "--rebase"], check=False)

        # Add all relevant changes
        subprocess.run(["git", "add", "models/"], check=False)
        subprocess.run(["git", "add", "mlops/last_metrics.json"], check=False)

        # Commit (no error if there is nothing to commit)
        subprocess.run(["git", "commit", "-m", message], check=False)

        # Push using the default remote
        subprocess.run(["git", "push"], check=False)

        print("⬆ Git commit & push attempted.")

    except Exception as exc:
        print(f"❌ git_commit_and_push failed: {exc}")
