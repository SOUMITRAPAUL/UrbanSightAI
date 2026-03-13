from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.data_generation import prepare_datasets


if __name__ == "__main__":
    prepare_datasets()
    print("Datasets prepared successfully.")
