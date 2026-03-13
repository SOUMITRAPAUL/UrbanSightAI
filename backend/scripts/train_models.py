from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.model_training import train_all_models


if __name__ == "__main__":
    metrics = train_all_models()
    print("Model training complete.")
    print(metrics)
