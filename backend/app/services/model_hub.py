from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.core.config import MODEL_DIR
from app.services.segmentation import MiniUNet


class ModelHub:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.prioritizer = None
        self.civic_classifier = None
        self.drain_monitor = None
        self.drain_threshold: float = 0.5
        self.flood_model = None
        self.flood_model_q15 = None
        self.flood_model_q85 = None
        self.segmentation_model: MiniUNet | None = None
        self.metrics: dict[str, float] = {}
        self.trust_report: dict[str, Any] = {}
        self.model_cards: dict[str, Any] = {}
        self.civic_review_threshold: float = 0.55

    def load(self) -> None:
        self.prioritizer = joblib.load(self.model_dir / "prioritizer.joblib")
        self.civic_classifier = joblib.load(self.model_dir / "civic_nlp.joblib")
        self.drain_monitor = joblib.load(self.model_dir / "drain_monitor.joblib")
        self.flood_model = joblib.load(self.model_dir / "flood_risk.joblib")
        q15_path = self.model_dir / "flood_risk_q15.joblib"
        q85_path = self.model_dir / "flood_risk_q85.joblib"
        self.flood_model_q15 = joblib.load(q15_path) if q15_path.exists() else None
        self.flood_model_q85 = joblib.load(q85_path) if q85_path.exists() else None

        self.segmentation_model = MiniUNet()
        state = torch.load(self.model_dir / "segmentation_unet.pt", map_location="cpu")
        self.segmentation_model.load_state_dict(state)
        self.segmentation_model.eval()

        metrics_path = self.model_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as file:
                self.metrics = json.load(file)

        trust_path = self.model_dir / "trust_report.json"
        if trust_path.exists():
            with trust_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                self.trust_report = payload

        card_path = self.model_dir / "model_cards.json"
        if card_path.exists():
            with card_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                self.model_cards = payload

        drain_meta = self.model_dir / "drain_monitor_meta.json"
        if drain_meta.exists():
            with drain_meta.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.drain_threshold = float(payload.get("classification_threshold", self.drain_threshold))

        civic_meta = self.model_dir / "civic_nlp_meta.json"
        if civic_meta.exists():
            with civic_meta.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.civic_review_threshold = float(payload.get("confidence_review_threshold", self.civic_review_threshold))

    def score_interventions(self, interventions: pd.DataFrame) -> np.ndarray:
        if self.prioritizer is None:
            raise RuntimeError("Prioritizer model is not loaded.")
        feature_cols = [
            "category",
            "agency",
            "permit_required",
            "estimated_cost_lakh",
            "expected_beneficiaries",
            "feasibility",
            "equity_need",
            "urgency",
        ]
        return self.prioritizer.predict(interventions[feature_cols])

    def predict_drain_blockage(self, segments: pd.DataFrame) -> np.ndarray:
        if self.drain_monitor is None:
            raise RuntimeError("Drain monitor model is not loaded.")
        feature_cols = [
            "segment_length_m",
            "water_proximity",
            "house_density",
            "citizen_pressure",
            "rainfall_sensor_mm",
            "pump_runtime_hours",
            "last_maintenance_days",
        ]
        frame = segments[feature_cols]
        if hasattr(self.drain_monitor, "predict_proba"):
            return self.drain_monitor.predict_proba(frame)[:, 1]
        return np.clip(self.drain_monitor.predict(frame), 0.0, 1.0)

    def predict_flood_risk(self, features: pd.DataFrame) -> np.ndarray:
        if self.flood_model is None:
            raise RuntimeError("Flood model is not loaded.")
        feature_cols = [
            "water_proximity",
            "drainage_congestion",
            "impervious_surface",
            "elevation_proxy",
            "rainfall_sensor_mm",
            "citizen_flood_pressure",
        ]
        pred = self.flood_model.predict(features[feature_cols])
        return np.clip(pred.astype(float), 0.0, 1.0)

    def predict_flood_risk_interval(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        base = self.predict_flood_risk(features)
        if self.flood_model_q15 is None or self.flood_model_q85 is None:
            return {"risk": base, "low": np.clip(base - 0.12, 0.0, 1.0), "high": np.clip(base + 0.12, 0.0, 1.0)}
        feature_cols = [
            "water_proximity",
            "drainage_congestion",
            "impervious_surface",
            "elevation_proxy",
            "rainfall_sensor_mm",
            "citizen_flood_pressure",
        ]
        low = np.clip(self.flood_model_q15.predict(features[feature_cols]).astype(float), 0.0, 1.0)
        high = np.clip(self.flood_model_q85.predict(features[feature_cols]).astype(float), 0.0, 1.0)
        lo = np.minimum(low, high)
        hi = np.maximum(low, high)
        return {"risk": base, "low": lo, "high": hi}

    def _sentiment_score(self, text: str) -> float:
        low = text.lower()
        negative = [
            "urgent",
            "broken",
            "flood",
            "waterlogging",
            "blocked",
            "clogged",
            "problem",
            "জরুরি",
            "সমস্যা",
            "বন্ধ",
            "দুর্গন্ধ",
            "জলাবদ্ধতা",
            "ভাঙা",
        ]
        positive = ["thanks", "resolved", "improved", "ধন্যবাদ", "সমাধান", "ভালো"]
        score = -0.4
        score -= sum(0.07 for word in negative if word in low)
        score += sum(0.06 for word in positive if word in low)
        return float(np.clip(score, -1.0, 1.0))

    def classify_civic_report(self, text: str) -> dict[str, Any]:
        if self.civic_classifier is None:
            raise RuntimeError("Civic NLP model is not loaded.")
        proba = self.civic_classifier.predict_proba([text])[0]
        labels = self.civic_classifier.classes_
        idx = int(np.argmax(proba))
        category = str(labels[idx])
        confidence = float(proba[idx])
        sentiment = self._sentiment_score(text)
        priority = float(
            np.clip(
                0.45
                + (0.25 if category in {"blocked_drain", "flooding"} else 0.1)
                + abs(min(sentiment, 0.0)) * 0.3,
                0.1,
                1.0,
            )
        )
        review_required = bool(confidence < self.civic_review_threshold)
        return {
            "category": category,
            "confidence": round(confidence, 4),
            "sentiment_score": round(sentiment, 4),
            "priority_weight": round(priority, 4),
            "review_required": review_required,
        }

    def infer_informal_area(self, images: np.ndarray) -> np.ndarray:
        if self.segmentation_model is None:
            raise RuntimeError("Segmentation model is not loaded.")
        tensor = torch.tensor(images, dtype=torch.float32)
        with torch.no_grad():
            logits = self.segmentation_model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > 0.5).astype(np.float32)
        return binary.mean(axis=(1, 2, 3))

    def get_trust_summary(self) -> dict[str, Any]:
        if self.trust_report:
            return self.trust_report
        return {
            "generated_at_utc": None,
            "overall_trust_index": float(self.metrics.get("overall_trust_index", 0.0)),
            "components": [],
            "monitoring_playbook": [],
            "note": "Trust report is not available. Re-run training to generate artifacts.",
        }


MODEL_HUB = ModelHub()
