from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "urbansightai.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
REAL_DATA_ONLY = os.getenv("URBANSIGHT_REAL_DATA_ONLY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "urbansightai-pilot-secret")
TOKEN_EXPIRES_MINUTES = 60 * 10
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama").strip()

WARD_BOUNDS_PATH = PROCESSED_DIR / "ward_boundaries.json"

DEFAULT_USERS = [
    {"username": "planner", "password": "pilot123", "role": "planner"},
    {"username": "enumerator", "password": "pilot123", "role": "enumerator"},
    {"username": "viewer", "password": "pilot123", "role": "viewer"},
]
