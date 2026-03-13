from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import MODEL_DIR, PROCESSED_DIR
from app.services.segmentation import MiniUNet


SEED = 42


def _float(value: Any) -> float:
    try:
        result = float(value)
        if np.isfinite(result):
            return result
        return 0.0
    except Exception:
        return 0.0


def _to_py(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _group_split_indices(
    n_rows: int,
    groups: pd.Series | np.ndarray | list[Any],
    test_size: float = 0.25,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    group_values = np.asarray(groups)
    if len(np.unique(group_values)) >= 4:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(idx, groups=group_values))
        return np.asarray(train_idx), np.asarray(test_idx)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 12) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * float(np.mean(mask))
    return float(ece)


def _ece_multiclass(
    y_true: np.ndarray,
    y_pred_labels: np.ndarray,
    y_confidence: np.ndarray,
    bins: int = 12,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    correct = (y_true == y_pred_labels).astype(np.float64)
    ece = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (y_confidence >= low) & (y_confidence <= high)
        else:
            mask = (y_confidence >= low) & (y_confidence < high)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(y_confidence[mask]))
        ece += abs(acc - conf) * float(np.mean(mask))
    return float(ece)


def _residual_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_calibration_residuals: np.ndarray,
    quantile: float = 0.9,
) -> dict[str, float]:
    q = float(np.quantile(abs_calibration_residuals, quantile)) if len(abs_calibration_residuals) else 0.0
    lower = y_pred - q
    upper = y_pred + q
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper))) if len(y_true) else 0.0
    return {
        "residual_quantile": q,
        "interval_coverage": coverage,
        "interval_avg_width": float(2.0 * q),
    }


def _top_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    if len(feature_names) == 0 or len(importances) == 0:
        return []
    top_idx = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": str(feature_names[int(i)]),
            "importance": round(_float(importances[int(i)]), 6),
        }
        for i in top_idx
    ]


def _train_prioritization_model(
    processed_dir: Path,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    interventions = pd.read_csv(processed_dir / "interventions.csv")
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
    target_col = "impact_signal"

    X = interventions[feature_cols].copy()
    y = interventions[target_col].astype(float).copy()
    groups = interventions["ward_id"] if "ward_id" in interventions.columns else pd.Series(np.arange(len(X)))

    train_idx, test_idx = _group_split_indices(len(X), groups=groups, test_size=0.25, random_state=SEED)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["category", "agency", "permit_required"],
            ),
            (
                "numeric",
                "passthrough",
                [
                    "estimated_cost_lakh",
                    "expected_beneficiaries",
                    "feasibility",
                    "equity_need",
                    "urgency",
                ],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", GradientBoostingRegressor(random_state=SEED)),
        ]
    )
    model.fit(X_fit, y_fit)

    cal_pred = model.predict(X_cal)
    cal_resid = np.abs(y_cal.to_numpy() - cal_pred)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = _rmse(y_test, pred)
    r2 = float(r2_score(y_test, pred))
    interval = _residual_interval(y_test.to_numpy(), pred, cal_resid, quantile=0.9)

    permit_mae: dict[str, float] = {}
    for group_value in [False, True]:
        mask = X_test["permit_required"].astype(bool) == group_value
        if int(mask.sum()) < 15:
            continue
        permit_mae[str(group_value)] = float(mean_absolute_error(y_test[mask], pred[mask]))
    permit_gap = (
        float(abs(permit_mae.get("True", permit_mae.get("False", 0.0)) - permit_mae.get("False", 0.0)))
        if permit_mae
        else 0.0
    )

    category_mae: dict[str, float] = {}
    for category, frame in X_test.groupby("category"):
        if len(frame) < 20:
            continue
        idx = frame.index.to_numpy()
        category_mae[str(category)] = float(mean_absolute_error(y_test.iloc[idx], pred[idx]))
    category_mae = dict(
        sorted(category_mae.items(), key=lambda x: x[1], reverse=True)[:8]
    )

    regressor = model.named_steps["regressor"]
    preprocess_step = model.named_steps["preprocess"]
    feature_names = [str(name) for name in preprocess_step.get_feature_names_out().tolist()]
    importances = getattr(regressor, "feature_importances_", np.array([], dtype=np.float64))
    top_features = _top_feature_importance(feature_names, np.asarray(importances), top_k=12)

    joblib.dump(model, model_dir / "prioritizer.joblib")
    with (model_dir / "prioritizer_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "prediction_interval": "split-conformal residual interval",
                "interval_quantile": 0.9,
                "residual_quantile_width": round(interval["residual_quantile"], 6),
                "group_holdout": "ward_id",
            },
            file,
            indent=2,
        )

    metrics = {
        "prioritizer_mae": round(mae, 6),
        "prioritizer_rmse": round(rmse, 6),
        "prioritizer_r2": round(r2, 6),
        "prioritizer_interval_coverage": round(interval["interval_coverage"], 6),
        "prioritizer_interval_avg_width": round(interval["interval_avg_width"], 6),
        "prioritizer_permit_mae_gap": round(permit_gap, 6),
    }
    card = {
        "model": "prioritizer",
        "algorithm": "GradientBoostingRegressor + one-hot features",
        "target": target_col,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols,
        "metrics": metrics,
        "top_features": top_features,
        "fairness_checks": {
            "permit_group_mae": {k: round(v, 6) for k, v in permit_mae.items()},
            "permit_group_mae_gap": round(permit_gap, 6),
            "category_mae_top": {k: round(v, 6) for k, v in category_mae.items()},
        },
        "uncertainty": {
            "method": "split conformal from calibration residuals",
            "interval_coverage_test": round(interval["interval_coverage"], 6),
            "interval_avg_width": round(interval["interval_avg_width"], 6),
        },
        "limitations": [
            "Trained on pilot-scale interventions and synthetic augmentation.",
            "Ranking confidence is weaker for unseen project categories/agencies.",
        ],
    }
    return metrics, card


def _train_nlp_model(
    processed_dir: Path,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_df = pd.read_csv(processed_dir / "civic_reports_train.csv")
    X = train_df["text"].astype(str)
    y = train_df["category"].astype(str)
    language = train_df["language"].astype(str)

    X_train, X_test, y_train, y_test, lang_train, lang_test = train_test_split(
        X,
        y,
        language,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    base_classifier = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=4500,
                    min_df=1,
                ),
            ),
            ("clf", LogisticRegression(max_iter=900, class_weight="balanced")),
        ]
    )
    classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        cv=3,
        method="sigmoid",
    )
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test)
    labels = [str(label) for label in classifier.classes_]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_true_idx = np.asarray([label_to_idx[str(lbl)] for lbl in y_test.to_numpy()], dtype=np.int64)
    pred_idx = np.argmax(proba, axis=1).astype(np.int64)
    top_conf = np.max(proba, axis=1)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))
    ece = _ece_multiclass(y_true_idx, pred_idx, top_conf, bins=12)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    per_class_recall = {
        label: round(_float(report.get(label, {}).get("recall", 0.0)), 6)
        for label in labels
    }

    language_macro_f1: dict[str, float] = {}
    for lang in sorted(lang_test.unique()):
        mask = lang_test == lang
        if int(mask.sum()) < 8:
            continue
        language_macro_f1[str(lang)] = float(
            f1_score(y_test[mask], pred[mask], average="macro", zero_division=0)
        )
    language_gap = (
        float(max(language_macro_f1.values()) - min(language_macro_f1.values()))
        if len(language_macro_f1) >= 2
        else 0.0
    )

    cm = confusion_matrix(y_test, pred, labels=labels)

    joblib.dump(classifier, model_dir / "civic_nlp.joblib")
    with (model_dir / "civic_nlp_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "labels": labels,
                "calibration": "sigmoid",
                "confidence_review_threshold": 0.55,
            },
            file,
            indent=2,
        )

    metrics = {
        "civic_classifier_accuracy": round(acc, 6),
        "civic_classifier_balanced_accuracy": round(bal_acc, 6),
        "civic_classifier_macro_f1": round(macro_f1, 6),
        "civic_classifier_weighted_f1": round(weighted_f1, 6),
        "civic_classifier_ece": round(ece, 6),
        "civic_language_f1_gap": round(language_gap, 6),
    }
    card = {
        "model": "civic_intelligence_nlp",
        "algorithm": "TF-IDF + LogisticRegression with calibrated probabilities",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "labels": labels,
        "metrics": metrics,
        "per_class_recall": per_class_recall,
        "language_macro_f1": {k: round(v, 6) for k, v in language_macro_f1.items()},
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[int(x) for x in row] for row in cm.tolist()],
        },
        "limitations": [
            "Bangla dialect and mixed-script colloquial reports can reduce confidence.",
            "Extremely short reports may require human validation.",
        ],
    }
    return metrics, card


def _train_drain_monitor_model(
    processed_dir: Path,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_df = pd.read_csv(processed_dir / "drainage_monitor_train.csv")
    feature_cols = [
        "segment_length_m",
        "water_proximity",
        "house_density",
        "citizen_pressure",
        "rainfall_sensor_mm",
        "pump_runtime_hours",
        "last_maintenance_days",
    ]
    X = train_df[feature_cols].astype(float).copy()
    y = train_df["blocked_target"].astype(int).copy()
    groups = train_df["ward_id"].astype(int) if "ward_id" in train_df.columns else pd.Series(np.arange(len(X)))

    if y.nunique() < 2:
        minority = 1 - int(y.iloc[0])
        synth = X.sample(n=min(120, len(X)), replace=True, random_state=SEED).copy()
        delta = 0.35 if minority == 1 else -0.35
        synth["water_proximity"] = np.clip(synth["water_proximity"] + delta, 0.0, 1.0)
        synth["citizen_pressure"] = np.clip(synth["citizen_pressure"] + delta * 0.7, 0.0, 1.0)
        synth["last_maintenance_days"] = np.clip(
            synth["last_maintenance_days"] + (45 if minority == 1 else -45),
            2.0,
            260.0,
        )
        X = pd.concat([X, synth], ignore_index=True)
        y = pd.concat([y, pd.Series([minority] * len(synth), dtype=int)], ignore_index=True)
        if isinstance(groups, pd.Series):
            groups = pd.concat(
                [groups, pd.Series([int(groups.iloc[0])] * len(synth), dtype=int)],
                ignore_index=True,
            )

    train_idx, test_idx = _group_split_indices(len(X), groups=groups, test_size=0.25, random_state=SEED)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    ward_test = (
        groups.iloc[test_idx].reset_index(drop=True)
        if isinstance(groups, pd.Series)
        else pd.Series(np.asarray(groups)[test_idx])
    )

    minority_label = int(y_train.value_counts().idxmin())
    minority_count = int((y_train == minority_label).sum())
    majority_count = int((y_train != minority_label).sum())
    desired_minority = max(180, min(520, int(majority_count * 0.12)))
    if minority_count < desired_minority:
        need = desired_minority - minority_count
        src = X_train[y_train == minority_label].copy()
        if src.empty:
            src = X_train.sample(n=min(240, len(X_train)), replace=True, random_state=SEED).copy()
            if minority_label == 0:
                src["water_proximity"] = np.clip(src["water_proximity"] - 0.42, 0.0, 1.0)
                src["citizen_pressure"] = np.clip(src["citizen_pressure"] - 0.35, 0.0, 1.0)
                src["rainfall_sensor_mm"] = np.clip(src["rainfall_sensor_mm"] - 12.0, 0.0, 160.0)
                src["pump_runtime_hours"] = np.clip(src["pump_runtime_hours"] - 1.3, 0.0, 24.0)
                src["last_maintenance_days"] = np.clip(src["last_maintenance_days"] - 55.0, 2.0, 320.0)
            else:
                src["water_proximity"] = np.clip(src["water_proximity"] + 0.35, 0.0, 1.0)
                src["citizen_pressure"] = np.clip(src["citizen_pressure"] + 0.30, 0.0, 1.0)
                src["rainfall_sensor_mm"] = np.clip(src["rainfall_sensor_mm"] + 15.0, 0.0, 160.0)
                src["pump_runtime_hours"] = np.clip(src["pump_runtime_hours"] + 1.6, 0.0, 24.0)
                src["last_maintenance_days"] = np.clip(src["last_maintenance_days"] + 65.0, 2.0, 320.0)
        synth = src.sample(n=need, replace=True, random_state=SEED).reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        noise = rng.normal(
            loc=0.0,
            scale=np.array([2.8, 0.05, 8.0, 0.05, 3.2, 0.45, 10.0], dtype=np.float64),
            size=(need, len(feature_cols)),
        )
        synth_values = synth.to_numpy(dtype=np.float64) + noise
        synth_df = pd.DataFrame(synth_values, columns=feature_cols)
        synth_df["segment_length_m"] = np.clip(synth_df["segment_length_m"], 4.0, 280.0)
        synth_df["water_proximity"] = np.clip(synth_df["water_proximity"], 0.0, 1.0)
        synth_df["house_density"] = np.clip(synth_df["house_density"], 0.0, 320.0)
        synth_df["citizen_pressure"] = np.clip(synth_df["citizen_pressure"], 0.0, 1.0)
        synth_df["rainfall_sensor_mm"] = np.clip(synth_df["rainfall_sensor_mm"], 0.0, 180.0)
        synth_df["pump_runtime_hours"] = np.clip(synth_df["pump_runtime_hours"], 0.0, 24.0)
        synth_df["last_maintenance_days"] = np.clip(synth_df["last_maintenance_days"], 2.0, 320.0)

        X_train = pd.concat([X_train, synth_df], ignore_index=True)
        y_train = pd.concat(
            [y_train, pd.Series([minority_label] * need, dtype=int)],
            ignore_index=True,
        )

    base_model = RandomForestClassifier(
        n_estimators=320,
        random_state=SEED,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(
        estimator=base_model,
        cv=3,
        method="sigmoid",
    )
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    threshold_candidates = np.linspace(0.25, 0.80, 56)
    best_threshold = 0.5
    best_f2 = -1.0
    for thr in threshold_candidates:
        train_pred = (train_proba >= thr).astype(int)
        score = float(fbeta_score(y_train, train_pred, beta=2.0, zero_division=0))
        if score > best_f2:
            best_f2 = score
            best_threshold = float(thr)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= best_threshold).astype(int)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = 0.5
    if not np.isfinite(auc):
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_test, proba))
    except Exception:
        pr_auc = 0.5
    if not np.isfinite(pr_auc):
        pr_auc = 0.5
    brier = float(brier_score_loss(y_test, proba))
    ece = _ece_binary(y_test.to_numpy(), proba, bins=12)

    ward_recall: dict[str, float] = {}
    for ward_id in sorted(int(v) for v in ward_test.unique().tolist()):
        mask = ward_test == ward_id
        if int(mask.sum()) < 18:
            continue
        if int(y_test[mask].sum()) == 0:
            continue
        ward_recall[str(int(ward_id))] = float(
            recall_score(y_test[mask], pred[mask], zero_division=0)
        )
    ward_recall_gap = (
        float(max(ward_recall.values()) - min(ward_recall.values()))
        if len(ward_recall) >= 2
        else 0.0
    )

    importance_model = RandomForestClassifier(
        n_estimators=220,
        random_state=SEED,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    importance_model.fit(X_train, y_train)
    top_features = _top_feature_importance(
        feature_cols,
        np.asarray(getattr(importance_model, "feature_importances_", np.zeros(len(feature_cols)))),
        top_k=len(feature_cols),
    )

    joblib.dump(model, model_dir / "drain_monitor.joblib")
    with (model_dir / "drain_monitor_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "classification_threshold": round(best_threshold, 6),
                "threshold_optimization": "maximize F2 on training split",
                "calibration": "sigmoid",
            },
            file,
            indent=2,
        )

    metrics = {
        "drain_monitor_accuracy": round(acc, 6),
        "drain_monitor_balanced_accuracy": round(bal_acc, 6),
        "drain_monitor_precision": round(precision, 6),
        "drain_monitor_recall": round(recall, 6),
        "drain_monitor_f1": round(f1, 6),
        "drain_monitor_auc": round(auc, 6),
        "drain_monitor_pr_auc": round(pr_auc, 6),
        "drain_monitor_brier": round(brier, 6),
        "drain_monitor_ece": round(ece, 6),
        "drain_monitor_ward_recall_gap": round(ward_recall_gap, 6),
        "drain_monitor_threshold": round(best_threshold, 6),
    }
    card = {
        "model": "drainage_blockage_monitor",
        "algorithm": "Calibrated RandomForestClassifier",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols,
        "metrics": metrics,
        "top_features": top_features,
        "ward_recall": {k: round(v, 6) for k, v in ward_recall.items()},
        "decision_threshold": round(best_threshold, 6),
        "limitations": [
            "Severe unseen weather anomalies can shift score calibration.",
            "Low-sensor wards rely more heavily on proxy features.",
        ],
    }
    return metrics, card


def _train_flood_risk_model(
    processed_dir: Path,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_df = pd.read_csv(processed_dir / "flood_risk_train.csv")
    feature_cols = [
        "water_proximity",
        "drainage_congestion",
        "impervious_surface",
        "elevation_proxy",
        "rainfall_sensor_mm",
        "citizen_flood_pressure",
    ]
    X = train_df[feature_cols].astype(float).copy()
    y = train_df["target_risk"].astype(float).clip(0.0, 1.0).copy()
    groups = train_df["ward_id"].astype(int) if "ward_id" in train_df.columns else pd.Series(np.arange(len(X)))

    train_idx, test_idx = _group_split_indices(len(X), groups=groups, test_size=0.25, random_state=SEED)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    base_for_interval = GradientBoostingRegressor(random_state=SEED)
    base_for_interval.fit(X_fit, y_fit)
    cal_pred = np.clip(base_for_interval.predict(X_cal), 0.0, 1.0)
    cal_resid = np.abs(y_cal.to_numpy() - cal_pred)

    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0.0, 1.0)

    lower_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=SEED)
    upper_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=SEED)
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)
    low_q = np.clip(lower_model.predict(X_test), 0.0, 1.0)
    high_q = np.clip(upper_model.predict(X_test), 0.0, 1.0)
    lo = np.minimum(low_q, high_q)
    hi = np.maximum(low_q, high_q)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = _rmse(y_test, pred)
    r2 = float(r2_score(y_test, pred))
    quantile_coverage = float(np.mean((y_test.to_numpy() >= lo) & (y_test.to_numpy() <= hi)))
    quantile_width = float(np.mean(hi - lo))

    conformal = _residual_interval(y_test.to_numpy(), pred, cal_resid, quantile=0.9)

    bins = pd.cut(
        y_test,
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["low", "medium", "high"],
    )
    bucket_mae: dict[str, float] = {}
    for bucket in ["low", "medium", "high"]:
        mask = bins == bucket
        if int(mask.sum()) < 20:
            continue
        bucket_mae[bucket] = float(mean_absolute_error(y_test[mask], pred[mask]))
    bucket_gap = (
        float(max(bucket_mae.values()) - min(bucket_mae.values()))
        if len(bucket_mae) >= 2
        else 0.0
    )

    top_features = _top_feature_importance(
        feature_cols,
        np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_cols)))),
        top_k=len(feature_cols),
    )

    joblib.dump(model, model_dir / "flood_risk.joblib")
    joblib.dump(lower_model, model_dir / "flood_risk_q15.joblib")
    joblib.dump(upper_model, model_dir / "flood_risk_q85.joblib")
    with (model_dir / "flood_risk_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "quantile_bounds": [0.1, 0.9],
                "conformal_interval_quantile": 0.9,
                "conformal_residual_width": round(conformal["residual_quantile"], 6),
            },
            file,
            indent=2,
        )

    metrics = {
        "flood_model_mae": round(mae, 6),
        "flood_model_rmse": round(rmse, 6),
        "flood_model_r2": round(r2, 6),
        "flood_model_quantile_coverage": round(quantile_coverage, 6),
        "flood_model_quantile_width": round(quantile_width, 6),
        "flood_model_conformal_coverage": round(conformal["interval_coverage"], 6),
        "flood_model_bucket_mae_gap": round(bucket_gap, 6),
    }
    card = {
        "model": "flood_risk_regressor",
        "algorithm": "GradientBoostingRegressor + quantile uncertainty heads",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols,
        "metrics": metrics,
        "top_features": top_features,
        "risk_bucket_mae": {k: round(v, 6) for k, v in bucket_mae.items()},
        "uncertainty": {
            "quantile_coverage": round(quantile_coverage, 6),
            "quantile_avg_width": round(quantile_width, 6),
            "conformal_coverage": round(conformal["interval_coverage"], 6),
            "conformal_avg_width": round(conformal["interval_avg_width"], 6),
        },
        "limitations": [
            "Relies on proxy elevation and congestion features, not full hydraulic simulation.",
            "Rare extreme rainfall periods may require recalibration.",
        ],
    }
    return metrics, card


def _train_segmentation_model(
    processed_dir: Path,
    model_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    data = np.load(processed_dir / "segmentation_data.npz")
    train_images = torch.tensor(data["train_images"], dtype=torch.float32)
    train_masks = torch.tensor(data["train_masks"], dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_masks,
        test_size=0.2,
        random_state=SEED,
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=24, shuffle=False)

    torch.manual_seed(SEED)
    model = MiniUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(6):
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    iou_scores: list[float] = []
    dice_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            intersection = (pred * batch_y).sum(dim=(1, 2, 3))
            pred_area = pred.sum(dim=(1, 2, 3))
            true_area = batch_y.sum(dim=(1, 2, 3))
            union = ((pred + batch_y) > 0).float().sum(dim=(1, 2, 3)) + 1e-7

            iou = intersection / union
            dice = (2 * intersection + 1e-7) / (pred_area + true_area + 1e-7)
            precision = (intersection + 1e-7) / (pred_area + 1e-7)
            recall = (intersection + 1e-7) / (true_area + 1e-7)

            iou_scores.extend(iou.cpu().numpy().tolist())
            dice_scores.extend(dice.cpu().numpy().tolist())
            precision_scores.extend(precision.cpu().numpy().tolist())
            recall_scores.extend(recall.cpu().numpy().tolist())

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mean_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    torch.save(model.state_dict(), model_dir / "segmentation_unet.pt")
    with (model_dir / "segmentation_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input_size": 64,
                "model": "MiniUNet",
                "epochs": 6,
                "threshold": 0.5,
            },
            file,
            indent=2,
        )

    metrics = {
        "segmentation_mean_iou": round(mean_iou, 6),
        "segmentation_mean_dice": round(mean_dice, 6),
        "segmentation_mean_precision": round(mean_precision, 6),
        "segmentation_mean_recall": round(mean_recall, 6),
    }
    card = {
        "model": "informal_area_segmentation",
        "algorithm": "MiniUNet",
        "training_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "metrics": metrics,
        "limitations": [
            "Pilot model uses synthetic-style training patches for rapid prototyping.",
            "Boundary precision may degrade on low-contrast or clouded imagery.",
        ],
    }
    return metrics, card


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _build_trust_report(
    metrics: dict[str, float],
    cards: dict[str, dict[str, Any]],
    dataset_version: str,
) -> dict[str, Any]:
    prior_score = _clip01(
        0.35 * (1.0 - metrics.get("prioritizer_mae", 1.0) / 0.16)
        + 0.25 * _clip01((metrics.get("prioritizer_r2", 0.0) + 1.0) / 2.0)
        + 0.20 * metrics.get("prioritizer_interval_coverage", 0.0)
        + 0.20 * (1.0 - min(metrics.get("prioritizer_permit_mae_gap", 0.2) / 0.12, 1.0))
    )
    civic_score = _clip01(
        0.42 * metrics.get("civic_classifier_macro_f1", 0.0)
        + 0.32 * metrics.get("civic_classifier_balanced_accuracy", 0.0)
        + 0.16 * (1.0 - min(metrics.get("civic_classifier_ece", 1.0), 1.0))
        + 0.10 * (1.0 - min(metrics.get("civic_language_f1_gap", 1.0), 1.0))
    )
    drain_score = _clip01(
        0.30 * metrics.get("drain_monitor_auc", 0.0)
        + 0.30 * metrics.get("drain_monitor_recall", 0.0)
        + 0.20 * metrics.get("drain_monitor_f1", 0.0)
        + 0.10 * (1.0 - min(metrics.get("drain_monitor_ece", 1.0), 1.0))
        + 0.10 * (1.0 - min(metrics.get("drain_monitor_ward_recall_gap", 1.0), 1.0))
    )
    flood_score = _clip01(
        0.35 * (1.0 - metrics.get("flood_model_mae", 1.0) / 0.20)
        + 0.20 * _clip01((metrics.get("flood_model_r2", 0.0) + 1.0) / 2.0)
        + 0.25 * metrics.get("flood_model_quantile_coverage", 0.0)
        + 0.20 * (1.0 - min(metrics.get("flood_model_bucket_mae_gap", 1.0), 1.0))
    )
    seg_score = _clip01(
        0.60 * metrics.get("segmentation_mean_iou", 0.0)
        + 0.40 * metrics.get("segmentation_mean_dice", 0.0)
    )

    overall = _clip01(
        0.26 * prior_score
        + 0.19 * civic_score
        + 0.21 * drain_score
        + 0.18 * flood_score
        + 0.16 * seg_score
    )

    def status(score: float) -> str:
        if score >= 0.78:
            return "good"
        if score >= 0.62:
            return "monitor"
        return "needs_retraining"

    components = [
        {
            "id": "prioritizer",
            "name": "Prioritization & Decision Ranker",
            "trust_score": round(prior_score, 4),
            "status": status(prior_score),
            "key_metrics": {
                "mae": round(metrics.get("prioritizer_mae", 0.0), 6),
                "r2": round(metrics.get("prioritizer_r2", 0.0), 6),
                "interval_coverage": round(metrics.get("prioritizer_interval_coverage", 0.0), 6),
                "permit_mae_gap": round(metrics.get("prioritizer_permit_mae_gap", 0.0), 6),
            },
            "strengths": [
                "Group holdout by ward to reduce spatial leakage.",
                "Uncertainty interval provided per score family.",
            ],
            "known_risks": cards.get("prioritizer", {}).get("limitations", []),
            "recommended_controls": [
                "Flag interventions with low rank margin for planner review.",
                "Retrain quarterly after field execution outcomes are logged.",
            ],
        },
        {
            "id": "civic_nlp",
            "name": "Civic Intelligence NLP",
            "trust_score": round(civic_score, 4),
            "status": status(civic_score),
            "key_metrics": {
                "macro_f1": round(metrics.get("civic_classifier_macro_f1", 0.0), 6),
                "balanced_accuracy": round(metrics.get("civic_classifier_balanced_accuracy", 0.0), 6),
                "ece": round(metrics.get("civic_classifier_ece", 0.0), 6),
                "language_f1_gap": round(metrics.get("civic_language_f1_gap", 0.0), 6),
            },
            "strengths": [
                "Calibrated classifier for confidence-aware routing.",
                "Per-language disparity checks included in training report.",
            ],
            "known_risks": cards.get("civic_nlp", {}).get("limitations", []),
            "recommended_controls": [
                "Route low-confidence reports to enumerator validation queue.",
                "Refresh language data with monthly citizen report samples.",
            ],
        },
        {
            "id": "drain_monitor",
            "name": "Drainage Blockage Monitor",
            "trust_score": round(drain_score, 4),
            "status": status(drain_score),
            "key_metrics": {
                "auc": round(metrics.get("drain_monitor_auc", 0.0), 6),
                "recall": round(metrics.get("drain_monitor_recall", 0.0), 6),
                "f1": round(metrics.get("drain_monitor_f1", 0.0), 6),
                "ward_recall_gap": round(metrics.get("drain_monitor_ward_recall_gap", 0.0), 6),
            },
            "strengths": [
                "Threshold tuned for recall-weighted early warning.",
                "Ward-level robustness gap measured explicitly.",
            ],
            "known_risks": cards.get("drain_monitor", {}).get("limitations", []),
            "recommended_controls": [
                "Prioritize site inspection for high-risk segments with sparse sensors.",
                "Recalibrate threshold during monsoon peak periods.",
            ],
        },
        {
            "id": "flood_risk",
            "name": "Flood Risk Regressor",
            "trust_score": round(flood_score, 4),
            "status": status(flood_score),
            "key_metrics": {
                "mae": round(metrics.get("flood_model_mae", 0.0), 6),
                "r2": round(metrics.get("flood_model_r2", 0.0), 6),
                "quantile_coverage": round(metrics.get("flood_model_quantile_coverage", 0.0), 6),
                "bucket_mae_gap": round(metrics.get("flood_model_bucket_mae_gap", 0.0), 6),
            },
            "strengths": [
                "Includes quantile uncertainty models for decision envelopes.",
                "Bucket-level error diagnostics across low/medium/high risk zones.",
            ],
            "known_risks": cards.get("flood_risk", {}).get("limitations", []),
            "recommended_controls": [
                "Trigger manual review when uncertainty band width exceeds threshold.",
                "Incorporate hydraulic simulation layers in future iterations.",
            ],
        },
        {
            "id": "segmentation",
            "name": "Informal Area Segmentation",
            "trust_score": round(seg_score, 4),
            "status": status(seg_score),
            "key_metrics": {
                "mean_iou": round(metrics.get("segmentation_mean_iou", 0.0), 6),
                "mean_dice": round(metrics.get("segmentation_mean_dice", 0.0), 6),
                "mean_precision": round(metrics.get("segmentation_mean_precision", 0.0), 6),
                "mean_recall": round(metrics.get("segmentation_mean_recall", 0.0), 6),
            },
            "strengths": [
                "Segmentation quality validated with IoU and Dice.",
                "Balanced precision/recall reporting for footprint extraction.",
            ],
            "known_risks": cards.get("segmentation", {}).get("limitations", []),
            "recommended_controls": [
                "Use field verification for uncertain boundary regions.",
                "Expand training set with seasonally diverse imagery.",
            ],
        },
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "overall_trust_index": round(overall, 6),
        "components": components,
        "monitoring_playbook": [
            "Recompute trust report after each live data refresh batch.",
            "Escalate components with status=needs_retraining to planner review.",
            "Track calibration drift and group disparity every retraining cycle.",
        ],
    }


def train_all_models(
    processed_dir: Path = PROCESSED_DIR,
    model_dir: Path = MODEL_DIR,
) -> dict[str, float]:
    model_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    metrics: dict[str, float] = {}
    cards: dict[str, dict[str, Any]] = {}

    part_metrics, card = _train_prioritization_model(processed_dir, model_dir)
    metrics.update(part_metrics)
    cards["prioritizer"] = card

    part_metrics, card = _train_nlp_model(processed_dir, model_dir)
    metrics.update(part_metrics)
    cards["civic_nlp"] = card

    part_metrics, card = _train_drain_monitor_model(processed_dir, model_dir)
    metrics.update(part_metrics)
    cards["drain_monitor"] = card

    part_metrics, card = _train_flood_risk_model(processed_dir, model_dir)
    metrics.update(part_metrics)
    cards["flood_risk"] = card

    part_metrics, card = _train_segmentation_model(processed_dir, model_dir)
    metrics.update(part_metrics)
    cards["segmentation"] = card

    dataset_version = "unknown"
    dataset_meta_path = processed_dir / "dataset_meta.json"
    if dataset_meta_path.exists():
        try:
            with dataset_meta_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            dataset_version = str(payload.get("dataset_version", "unknown"))
        except Exception:
            dataset_version = "unknown"

    trust_report = _build_trust_report(metrics, cards, dataset_version)
    metrics["overall_trust_index"] = round(
        _float(trust_report.get("overall_trust_index", 0.0)),
        6,
    )

    normalized_metrics = {
        key: round(_float(value), 6)
        for key, value in metrics.items()
    }
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(normalized_metrics, file, indent=2)

    with (model_dir / "model_cards.json").open("w", encoding="utf-8") as file:
        json.dump(cards, file, indent=2, default=_to_py)

    with (model_dir / "trust_report.json").open("w", encoding="utf-8") as file:
        json.dump(trust_report, file, indent=2, default=_to_py)

    with (model_dir / "model_meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_version": dataset_version,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "overall_trust_index": normalized_metrics["overall_trust_index"],
                "artifacts": [
                    "metrics.json",
                    "model_cards.json",
                    "trust_report.json",
                ],
            },
            file,
            indent=2,
        )

    return normalized_metrics
