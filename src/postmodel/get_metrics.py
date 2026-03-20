from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

def get_metrics():
    """
    Load all of the modeling JSON files
    """
    print("MODEL_DIR:", MODEL_DIR)
    metrics = {}
    for file in MODEL_DIR.glob("*metrics.json"):
        with open(file, "r") as f:
            metrics[file.stem] = json.load(f)
    print(f"Loaded {len(metrics)} metrics files just now.")
    return metrics 
