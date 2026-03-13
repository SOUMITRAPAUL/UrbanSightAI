from __future__ import annotations

from collections import Counter
import json
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_roles
from app.core.config import PROCESSED_DIR
from app.db import get_db
from app.models import CitizenReport, Intervention, User, Ward, WardIndicator
from app.schemas import (
    AIComponentStatus,
    AuditTrailItem,
    ConsultRequest,
    ConsultResponse,
    InterAgencyPacket,
    InterAgencyTask,
    LiveIngestRequest,
    LiveSourceEvent,
    LiveUpdateResponse,
    NotificationFeedItem,
    PlanningStrategy,
    PublicDataSources,
    SDG11GovernanceCard,
    SDG11TargetStatus,
    ScenarioResult,
    TwinAction,
    TwinAreaAsset,
    TwinBlockedDrainSegment,
    TwinHotspot,
    TwinHouse,
    TwinLayers,
    TwinProblem,
    TopWorkItem,
    TopWorklistResponse,
    LiveWeatherSnapshot,
    WardDigitalTwin,
    WardIndicatorResponse,
    WardSummary,
    WardTwinSceneResponse,
    WardWorkflowResponse,
    WorkflowCandidate,
    WorkflowSourceRecord,
    WorkflowStageStatus,
)
from app.services.audit import list_audit_events, log_audit_event
from app.services.model_hub import MODEL_HUB
from app.services.public_data import fetch_open_meteo_current
from app.services.rag_service import RAG_SERVICE
from app.services.scenario import (
    build_counterfactuals,
    predict_budget_allocation,
    simulate_policy_scenario,
)
from app.services.ward_metrics import (
    estimate_blocked_network_scale,
    estimate_exposed_population,
    score_live_flood_risk,
    score_sdg11,
    summarize_ward_morphology,
)


router = APIRouter(prefix="/api", tags=["dashboard"])


def _load_json_file(path: str) -> dict[str, dict[str, int] | list[list[float]]]:
    with (PROCESSED_DIR / path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_dataset_meta() -> dict[str, object]:
    path = PROCESSED_DIR / "dataset_meta.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _boundary_centroid(boundary: list[list[float]]) -> tuple[float, float]:
    if not boundary:
        return 23.8103, 90.4125
    lats = [float(pt[0]) for pt in boundary if isinstance(pt, list) and len(pt) == 2]
    lons = [float(pt[1]) for pt in boundary if isinstance(pt, list) and len(pt) == 2]
    if not lats or not lons:
        return 23.8103, 90.4125
    return float(np.mean(lats)), float(np.mean(lons))


def _weather_rainfall_mm(weather: dict[str, object]) -> float | None:
    precipitation = weather.get("precipitation_mm")
    rain = weather.get("rain_mm")
    values = [
        float(value)
        for value in (precipitation, rain)
        if isinstance(value, (int, float))
    ]
    if not values:
        return None
    return float(max(values))


def _public_source_payload(
    *,
    dataset_meta: dict[str, object],
    weather: dict[str, object],
    river_segments: int,
) -> PublicDataSources:
    return PublicDataSources(
        boundaries_source=str(dataset_meta.get("boundary_source", "openstreetmap_overpass_api")),
        map_features_source=str(dataset_meta.get("map_source", "openstreetmap_overpass_api")),
        weather_source=str(weather.get("source", "open-meteo")),
        weather_status=str(weather.get("status", "unknown")),
        weather_observed_at=weather.get("observed_at")
        if isinstance(weather.get("observed_at"), datetime)
        else None,
        rainfall_mm=_weather_rainfall_mm(weather),
        temperature_c=(
            float(weather["temperature_c"])
            if isinstance(weather.get("temperature_c"), (int, float))
            else None
        ),
        river_segments=river_segments,
        generated_at=datetime.now(timezone.utc),
    )


def _safe_list_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _rendered_layer_summary(
    *,
    source_summary: dict[str, object] | None,
    roads: object,
    drains: object,
    rivers: object,
    waterbodies: object,
    houses: object,
    playgrounds: object,
    parks: object,
    blocked_drain_network: object,
    blocked_drains: object,
    flood_zones: object,
    informal_zones: object,
) -> tuple[dict[str, int], list[str]]:
    rendered = {
        "roads": _safe_list_length(roads),
        "drains": _safe_list_length(drains),
        "rivers": _safe_list_length(rivers),
        "waterbodies": _safe_list_length(waterbodies),
        "houses": _safe_list_length(houses),
        "playgrounds": _safe_list_length(playgrounds),
        "parks": _safe_list_length(parks),
        "blocked_drain_network": _safe_list_length(blocked_drain_network),
        "blocked_drains": _safe_list_length(blocked_drains),
        "flood_zones": _safe_list_length(flood_zones),
        "informal_zones": _safe_list_length(informal_zones),
    }

    if not isinstance(source_summary, dict):
        return rendered, []

    deltas: list[str] = []
    for key in ["roads", "drains", "rivers", "waterbodies", "houses", "playgrounds", "parks"]:
        source_value = source_summary.get(key)
        if not isinstance(source_value, (int, float)):
            continue
        source_count = int(source_value)
        rendered_count = rendered[key]
        if source_count != rendered_count:
            deltas.append(f"{key.replace('_', ' ')} {source_count}->{rendered_count}")

    if not deltas:
        return rendered, []

    preview = ", ".join(deltas[:4])
    if len(deltas) > 4:
        preview = f"{preview}, +{len(deltas) - 4} more"
    return rendered, [
        f"Rendered layer counts override stale source summary values ({preview})."
    ]


def _meta_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _segmentation_inventory() -> tuple[int, int]:
    path = PROCESSED_DIR / "segmentation_data.npz"
    if not path.exists():
        return 0, 0
    try:
        with np.load(path) as data:
            train_tiles = int(len(data["train_images"])) if "train_images" in data else 0
            ward_tiles = int(len(data["ward_images"])) if "ward_images" in data else 0
        return train_tiles, ward_tiles
    except Exception:
        return 0, 0


def _workflow_source_inventory(
    *,
    ward: Ward,
    dataset_meta: dict[str, object],
    layer_summary: dict[str, int],
    interventions: list[Intervention],
    report_rows: list[CitizenReport],
    live_weather: dict[str, object],
) -> list[WorkflowSourceRecord]:
    dataset_updated = _meta_datetime(dataset_meta.get("generated_at_utc"))
    latest_report_time = max((row.created_at for row in report_rows), default=None)
    asset_records = int(
        sum(
            int(layer_summary.get(key, 0))
            for key in ["houses", "playgrounds", "parks", "roads", "drains", "waterbodies", "rivers"]
        )
    )
    segmentation_train_tiles, ward_tiles = _segmentation_inventory()
    return [
        WorkflowSourceRecord(
            name="Satellite / imagery tiles",
            kind="satellite",
            status="implemented" if ward_tiles > 0 else "partial",
            records=max(ward_tiles, segmentation_train_tiles),
            provenance="segmentation_data.npz (pilot imagery tiles)",
            last_updated=dataset_updated,
            notes="Ward imagery tiles available for informal-area segmentation.",
        ),
        WorkflowSourceRecord(
            name="Ward GIS boundaries and vectors",
            kind="gis",
            status="implemented",
            records=int(sum(int(layer_summary.get(key, 0)) for key in ["roads", "drains", "rivers", "waterbodies"])),
            provenance=(
                f"boundaries={dataset_meta.get('boundary_source', 'unknown')}; "
                f"map={dataset_meta.get('map_source', 'unknown')}"
            ),
            last_updated=dataset_updated,
            notes="Ward boundary, drain, river, road, and waterbody layers aligned to the same ward geometry.",
        ),
        WorkflowSourceRecord(
            name="Neighborhood asset register",
            kind="asset_register",
            status="implemented" if asset_records > 0 else "partial",
            records=asset_records,
            provenance=str(dataset_meta.get("map_source", "openstreetmap_overpass_api")),
            last_updated=dataset_updated,
            notes="Asset proxy built from public map features: houses, parks, playgrounds, and civic infrastructure.",
        ),
        WorkflowSourceRecord(
            name="Population / census baseline",
            kind="census_gridded_pop",
            status="implemented",
            records=int(ward.population),
            provenance="wards.csv population baseline with household counts",
            last_updated=dataset_updated,
            notes="Ward population and household baseline used for exposed-population and beneficiary estimation.",
        ),
        WorkflowSourceRecord(
            name="Past workorders / intervention register",
            kind="workorders",
            status="implemented" if interventions else "missing",
            records=len(interventions),
            provenance="interventions table (candidate micro-works and implementation pipeline)",
            last_updated=ward.last_updated,
            notes="Existing intervention library acts as the operational work-order candidate set.",
        ),
        WorkflowSourceRecord(
            name="Citizen and field reports",
            kind="citizen_reports",
            status="implemented" if report_rows else "partial",
            records=len(report_rows),
            provenance="citizen_reports table + ODK/KoBo submissions",
            last_updated=latest_report_time,
            notes="Bangla/English complaints and enumerator reports feed civic validation and ranking nudges.",
        ),
        WorkflowSourceRecord(
            name="Weather / sensor pulse stream",
            kind="weather_sensor",
            status="implemented" if str(live_weather.get("status", "unavailable")) != "unavailable" else "partial",
            records=1 if str(live_weather.get("status", "unavailable")) != "unavailable" else 0,
            provenance=str(live_weather.get("source", "open-meteo")),
            last_updated=live_weather.get("observed_at") if isinstance(live_weather.get("observed_at"), datetime) else None,
            notes="Live weather is fused with simulated sensor pressure during live-ingest model reruns.",
        ),
    ]


def _workflow_candidate_proposals(
    *,
    ward: Ward,
    indicator: WardIndicator,
    layer_summary: dict[str, int],
    report_rows: list[CitizenReport],
    ranked_items: list[TopWorkItem],
) -> list[WorkflowCandidate]:
    complaint_counts = Counter(row.category for row in report_rows)
    total_reports = max(len(report_rows), 1)
    blocked_pressure = float(
        (complaint_counts.get("blocked_drain", 0) + complaint_counts.get("flooding", 0))
        / total_reports
    )
    waste_pressure = float(complaint_counts.get("waste", 0) / total_reports)
    road_pressure = float(complaint_counts.get("road_damage", 0) / total_reports)
    water_pressure = float(complaint_counts.get("water_supply", 0) / total_reports)
    flood_risk = float(indicator.flood_risk_index)
    blocked_norm = float(np.clip(indicator.blocked_drain_count / 55.0, 0.0, 1.0))
    green_deficit = float(indicator.green_deficit_index)
    informal_ratio = float(indicator.informal_area_pct / 100.0)
    exposed_ratio = float(np.clip(indicator.exposed_population / max(ward.population, 1), 0.0, 1.0))
    river_count = int(layer_summary.get("rivers", 0) + layer_summary.get("waterbodies", 0))
    road_count = int(layer_summary.get("roads", 0))
    house_count = int(layer_summary.get("houses", 0))

    ranked_by_category: dict[str, TopWorkItem] = {}
    for item in ranked_items:
        ranked_by_category.setdefault(item.category, item)

    def expected_people(ratio: float) -> int:
        return int(np.clip(round(ward.population * ratio), 120, ward.population))

    proposals = [
        {
            "priority": 0.44 * blocked_norm + 0.34 * flood_risk + 0.22 * blocked_pressure,
            "title": "Drain desilting and grill-clearing bundle",
            "category": "Drainage",
            "agency": "City Corporation",
            "permit_required": False,
            "rough_cost_lakh": float(np.clip(0.55 + indicator.blocked_drain_count * 0.028 + flood_risk * 0.85, 0.45, 3.2)),
            "expected_beneficiaries": expected_people(0.06 + flood_risk * 0.10 + blocked_norm * 0.08),
            "trigger_metric": (
                f"blocked_drain_count={indicator.blocked_drain_count}, "
                f"flood_risk_index={indicator.flood_risk_index:.2f}, "
                f"blocked_report_share={blocked_pressure:.2f}"
            ),
            "rationale": "Generated from drainage blockage signals, live flood pressure, and citizen drain/flood complaints.",
            "evidence_sources": ["drain_monitor_model", "citizen_reports", "ward_drain_gis"],
        },
        {
            "priority": 0.46 * flood_risk + 0.28 * water_pressure + 0.26 * min(river_count / 18.0, 1.0),
            "title": "Canal edge desilting and outfall reopening",
            "category": "Water",
            "agency": "WASA",
            "permit_required": True,
            "rough_cost_lakh": float(np.clip(0.95 + flood_risk * 1.65 + river_count * 0.03, 0.8, 3.9)),
            "expected_beneficiaries": expected_people(0.05 + flood_risk * 0.11 + water_pressure * 0.05),
            "trigger_metric": (
                f"river_waterbody_segments={river_count}, "
                f"flood_risk_index={indicator.flood_risk_index:.2f}, "
                f"water_supply_report_share={water_pressure:.2f}"
            ),
            "rationale": "Auto-proposed where water-edge exposure and flood intensity imply blocked outfalls or canal-edge maintenance needs.",
            "evidence_sources": ["flood_risk_model", "waterbody_layers", "open_meteo"],
        },
        {
            "priority": 0.42 * waste_pressure + 0.28 * informal_ratio + 0.30 * min(house_count / 2600.0, 1.0),
            "title": "Waste hotspot clearance and container placement",
            "category": "Waste",
            "agency": "City Corporation",
            "permit_required": False,
            "rough_cost_lakh": float(np.clip(0.35 + waste_pressure * 1.4 + informal_ratio * 1.1, 0.25, 2.1)),
            "expected_beneficiaries": expected_people(0.04 + waste_pressure * 0.09 + informal_ratio * 0.06),
            "trigger_metric": (
                f"waste_report_share={waste_pressure:.2f}, "
                f"informal_area_pct={indicator.informal_area_pct:.1f}, "
                f"house_count={house_count}"
            ),
            "rationale": "Generated from waste complaints, dense residential fabric, and informal-settlement service gaps.",
            "evidence_sources": ["citizen_reports", "house_density_layer", "informal_area_segmentation"],
        },
        {
            "priority": 0.38 * road_pressure + 0.32 * exposed_ratio + 0.30 * min(road_count / 220.0, 1.0),
            "title": "Road patching and access restoration micro-works",
            "category": "Road",
            "agency": "LGED",
            "permit_required": False,
            "rough_cost_lakh": float(np.clip(0.6 + road_pressure * 1.8 + exposed_ratio * 0.9, 0.5, 3.0)),
            "expected_beneficiaries": expected_people(0.04 + exposed_ratio * 0.08 + road_pressure * 0.05),
            "trigger_metric": (
                f"road_damage_report_share={road_pressure:.2f}, "
                f"exposed_population={indicator.exposed_population}, "
                f"road_segments={road_count}"
            ),
            "rationale": "Auto-proposed for access bottlenecks affecting exposed households and reported road-damage clusters.",
            "evidence_sources": ["citizen_reports", "road_network_layer", "exposed_population_indicator"],
        },
        {
            "priority": 0.60 * green_deficit + 0.24 * informal_ratio + 0.16 * exposed_ratio,
            "title": "Pocket park and shade-corridor installation",
            "category": "Green",
            "agency": "RAJUK",
            "permit_required": True,
            "rough_cost_lakh": float(np.clip(0.7 + green_deficit * 1.5 + informal_ratio * 0.5, 0.5, 2.8)),
            "expected_beneficiaries": expected_people(0.03 + green_deficit * 0.07 + informal_ratio * 0.03),
            "trigger_metric": (
                f"green_deficit_index={indicator.green_deficit_index:.2f}, "
                f"informal_area_pct={indicator.informal_area_pct:.1f}, "
                f"exposed_ratio={exposed_ratio:.2f}"
            ),
            "rationale": "Generated from green-access deficit and equity pressure in dense low-amenity blocks.",
            "evidence_sources": ["green_deficit_indicator", "informal_area_segmentation", "ward_asset_layers"],
        },
        {
            "priority": 0.48 * informal_ratio + 0.32 * blocked_pressure + 0.20 * min(road_count / 220.0, 1.0),
            "title": "Community safety lighting and lane audit",
            "category": "Public Safety",
            "agency": "City Corporation",
            "permit_required": False,
            "rough_cost_lakh": float(np.clip(0.32 + informal_ratio * 1.15 + blocked_pressure * 0.6, 0.3, 1.8)),
            "expected_beneficiaries": expected_people(0.03 + informal_ratio * 0.08),
            "trigger_metric": (
                f"informal_area_pct={indicator.informal_area_pct:.1f}, "
                f"blocked_report_share={blocked_pressure:.2f}, "
                f"road_segments={road_count}"
            ),
            "rationale": "Generated where informal-settlement density and reported service breakdown imply higher night-time safety and access risk.",
            "evidence_sources": ["informal_area_segmentation", "citizen_reports", "road_network_layer"],
        },
    ]

    ordered = sorted(proposals, key=lambda item: float(item["priority"]), reverse=True)[:6]
    results: list[WorkflowCandidate] = []
    for item in ordered:
        mapped = ranked_by_category.get(str(item["category"]))
        results.append(
            WorkflowCandidate(
                title=str(item["title"]),
                category=str(item["category"]),
                agency=str(item["agency"]),
                permit_required=bool(item["permit_required"]),
                rough_cost_lakh=round(float(item["rough_cost_lakh"]), 3),
                expected_beneficiaries=int(item["expected_beneficiaries"]),
                trigger_metric=str(item["trigger_metric"]),
                rationale=str(item["rationale"]),
                evidence_sources=[str(source) for source in item["evidence_sources"]],
                mapped_intervention_id=mapped.id if mapped else None,
                mapped_intervention_title=mapped.title if mapped else None,
                mapped_ranking_score=round(float(mapped.ranking_score), 6) if mapped else None,
            )
        )
    return results


def _to_work_item(item: Intervention) -> TopWorkItem:
    return TopWorkItem(
        id=item.id,
        title=item.title,
        category=item.category,
        agency=item.agency,
        permit_required=item.permit_required,
        estimated_cost_lakh=item.estimated_cost_lakh,
        expected_beneficiaries=item.expected_beneficiaries,
        beneficiary_ci_low=item.beneficiary_ci_low,
        beneficiary_ci_high=item.beneficiary_ci_high,
        beneficiary_method=item.beneficiary_method,
        impact_per_lakh=item.impact_per_lakh,
        ranking_score=item.ranking_score,
        feasibility=item.feasibility,
        equity_need=item.equity_need,
        urgency=item.urgency,
        justification=item.justification,
    )


def _severity(value: float, medium: float, high: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _line_midpoint(path: list[list[float]]) -> tuple[float, float]:
    if not path:
        return 0.0, 0.0
    lat = sum(pt[0] for pt in path) / len(path)
    lon = sum(pt[1] for pt in path) / len(path)
    return float(lat), float(lon)


def _path_length_m(path: list[list[float]]) -> float:
    if not isinstance(path, list) or len(path) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(path) - 1):
        p1 = path[idx]
        p2 = path[idx + 1]
        if not (
            isinstance(p1, list)
            and isinstance(p2, list)
            and len(p1) == 2
            and len(p2) == 2
        ):
            continue
        lat_scale = 111_000.0
        lon_scale = 111_000.0 * float(np.cos(np.radians((float(p1[0]) + float(p2[0])) * 0.5)))
        d_lat = (float(p2[0]) - float(p1[0])) * lat_scale
        d_lon = (float(p2[1]) - float(p1[1])) * lon_scale
        total += float((d_lat**2 + d_lon**2) ** 0.5)
    return float(total)


def _drain_edge_capacity(paths: list[list[list[float]]]) -> int:
    total = 0
    for path in paths:
        if isinstance(path, list) and len(path) >= 2:
            total += max(len(path) - 1, 0)
    return int(total)


def _offset_polyline(path: list[list[float]], offset_deg: float) -> list[list[float]]:
    if not isinstance(path, list) or len(path) < 2:
        return []
    shifted: list[list[float]] = []
    for idx, point in enumerate(path):
        if not isinstance(point, list) or len(point) != 2:
            continue
        prev_pt = path[max(0, idx - 1)]
        next_pt = path[min(len(path) - 1, idx + 1)]
        if not (
            isinstance(prev_pt, list)
            and isinstance(next_pt, list)
            and len(prev_pt) == 2
            and len(next_pt) == 2
        ):
            shifted.append([float(point[0]), float(point[1])])
            continue
        d_lat = float(next_pt[0]) - float(prev_pt[0])
        d_lon = float(next_pt[1]) - float(prev_pt[1])
        norm = float((d_lat**2 + d_lon**2) ** 0.5)
        if norm < 1e-9:
            shifted.append([float(point[0]), float(point[1])])
            continue
        normal_lat = -d_lon / norm
        normal_lon = d_lat / norm
        shifted.append(
            [
                round(float(point[0]) + normal_lat * offset_deg, 6),
                round(float(point[1]) + normal_lon * offset_deg, 6),
            ]
        )
    return shifted


def _infer_drain_paths(
    layers: dict[str, object],
    indicator: WardIndicator,
) -> list[list[list[float]]]:
    roads = layers.get("roads", [])
    waterbodies = layers.get("waterbodies", [])
    rivers = layers.get("rivers", [])
    houses = layers.get("houses", [])
    if not isinstance(roads, list):
        roads = []
    if not isinstance(waterbodies, list):
        waterbodies = []
    if not isinstance(rivers, list):
        rivers = []
    if not isinstance(houses, list):
        houses = []
    if not roads:
        return []

    water_midpoints = [
        _line_midpoint(path)
        for path in [*waterbodies, *rivers]
        if isinstance(path, list) and len(path) >= 2
    ]
    candidates: list[dict[str, object]] = []
    for idx, path in enumerate(roads):
        if not isinstance(path, list) or len(path) < 2:
            continue
        length_m = _path_length_m(path)
        if length_m < 10.0:
            continue
        lat, lon = _line_midpoint(path)
        if water_midpoints:
            near_water = min(abs(lat - w_lat) + abs(lon - w_lon) for w_lat, w_lon in water_midpoints)
        else:
            near_water = 0.05
        water_proximity = float(np.clip(1.0 - near_water / 0.08, 0.0, 1.0))
        house_density = float(np.clip(_nearby_house_count(houses, lat, lon, radius=0.0029) / 26.0, 0.0, 1.0))
        length_score = float(np.clip(length_m / 180.0, 0.0, 1.0))
        score = (
            0.36 * water_proximity
            + 0.28 * house_density
            + 0.22 * length_score
            + 0.14 * float(np.clip(indicator.flood_risk_index, 0.0, 1.0))
        )
        candidates.append({"path": path, "score": score, "idx": idx})

    if not candidates:
        return []

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    target_paths = min(
        len(candidates),
        max(16, min(140, int(indicator.blocked_drain_count / 2) + 14)),
    )
    inferred: list[list[list[float]]] = []
    for pos, candidate in enumerate(candidates[:target_paths]):
        path = candidate["path"]
        if not isinstance(path, list):
            continue
        offset = 0.000028 if pos % 2 == 0 else -0.000028
        shifted = _offset_polyline(path, offset_deg=offset)
        if len(shifted) >= 2:
            inferred.append(shifted)
    return inferred


def _resolve_drain_geometry(
    layers: dict[str, object],
    indicator: WardIndicator,
) -> tuple[list[list[list[float]]], int, list[str]]:
    raw_paths = layers.get("drains", [])
    drain_paths = [path for path in raw_paths if isinstance(path, list) and len(path) >= 2] if isinstance(raw_paths, list) else []
    notes: list[str] = []
    required_edges = max(int(indicator.blocked_drain_count), 0)
    capacity = _drain_edge_capacity(drain_paths)

    if required_edges > 0 and capacity < max(required_edges, 12):
        inferred = _infer_drain_paths(layers, indicator)
        if inferred:
            if not drain_paths:
                notes.append(
                    "Public drain geometry was missing for this ward; inferred roadside drainage alignments are rendered from the road network."
                )
                drain_paths = inferred
            else:
                notes.append(
                    "Public drain geometry was sparse for this ward; inferred roadside drainage alignments were added for consistency."
                )
                drain_paths = [*drain_paths, *inferred]
            capacity = _drain_edge_capacity(drain_paths)

    effective_blocked_count = int(min(required_edges, capacity))
    if required_edges > effective_blocked_count:
        notes.append(
            f"Blocked-drain display count was clipped from {required_edges} to {effective_blocked_count} because the available drain geometry supports only {effective_blocked_count} renderable segments."
        )
    return drain_paths, effective_blocked_count, notes


def _build_actions(interventions: list[Intervention]) -> list[TwinAction]:
    ordered = sorted(interventions, key=lambda x: x.ranking_score, reverse=True)[:6]
    statuses = ["completed", "completed", "in_progress", "in_progress", "planned", "planned"]
    progress = [100, 100, 72, 64, 25, 15]

    actions: list[TwinAction] = []
    for idx, intervention in enumerate(ordered):
        actions.append(
            TwinAction(
                intervention_id=intervention.id,
                title=intervention.title,
                category=intervention.category,
                agency=intervention.agency,
                status=statuses[idx],
                progress_pct=progress[idx],
                estimated_cost_lakh=intervention.estimated_cost_lakh,
                expected_beneficiaries=intervention.expected_beneficiaries,
            )
        )
    return actions


def _build_problems(
    indicator: WardIndicator,
    report_rows: list[CitizenReport],
    blocked_drain_count: int | None = None,
) -> list[TwinProblem]:
    report_counter = Counter(row.category for row in report_rows)
    top_cat, top_count = ("none", 0) if not report_counter else report_counter.most_common(1)[0]
    blocked_count = (
        int(blocked_drain_count)
        if isinstance(blocked_drain_count, int)
        else int(indicator.blocked_drain_count)
    )

    problems = [
        TwinProblem(
            issue="Blocked Drain Network",
            severity=_severity(float(blocked_count), 8, 20, 35),
            value=float(blocked_count),
            summary=f"{blocked_count} potential blocked drain segments.",
        ),
        TwinProblem(
            issue="Flood Exposure Risk",
            severity=_severity(float(indicator.flood_risk_index), 0.08, 0.14, 0.22),
            value=float(indicator.flood_risk_index),
            summary=f"Flood risk index {indicator.flood_risk_index:.2f} for current ward surface profile.",
        ),
        TwinProblem(
            issue="Green Deficit",
            severity=_severity(float(indicator.green_deficit_index), 0.55, 0.72, 0.86),
            value=float(indicator.green_deficit_index),
            summary=f"Green deficit index {indicator.green_deficit_index:.2f}; low accessible green assets.",
        ),
        TwinProblem(
            issue="Citizen Complaint Pressure",
            severity=_severity(float(top_count), 3, 6, 10),
            value=float(top_count),
            summary=f"Top civic signal: {top_cat.replace('_', ' ')} ({top_count} recent reports).",
        ),
    ]
    return problems


def _build_hotspots(
    layers: dict[str, object],
    indicator: WardIndicator,
    report_rows: list[CitizenReport],
    blocked_drain_count: int | None = None,
) -> list[TwinHotspot]:
    hotspots: list[TwinHotspot] = []
    report_counter = Counter(row.category for row in report_rows)
    blocked_count = (
        int(blocked_drain_count)
        if isinstance(blocked_drain_count, int)
        else int(indicator.blocked_drain_count)
    )

    drain_lines = layers.get("drains", [])
    if isinstance(drain_lines, list):
        drain_severity = _severity(float(blocked_count), 8, 20, 35)
        for line in drain_lines[:3]:
            if not isinstance(line, list):
                continue
            lat, lon = _line_midpoint(line)
            hotspots.append(
                TwinHotspot(
                    lat=lat,
                    lon=lon,
                    issue="Blocked Drain Hotspot",
                    severity=drain_severity,
                    source="drain-network",
                )
            )

    waterbodies = layers.get("waterbodies", [])
    rivers = layers.get("rivers", [])
    if not isinstance(waterbodies, list):
        waterbodies = []
    if not isinstance(rivers, list):
        rivers = []
    water_lines = [*waterbodies, *rivers]
    if isinstance(water_lines, list):
        flood_severity = _severity(float(indicator.flood_risk_index), 0.08, 0.14, 0.22)
        for line in water_lines[:2]:
            if not isinstance(line, list):
                continue
            lat, lon = _line_midpoint(line)
            hotspots.append(
                TwinHotspot(
                    lat=lat,
                    lon=lon,
                    issue="Flood-prone Edge",
                    severity=flood_severity,
                    source="water-proximity",
                )
            )

    houses = layers.get("houses", [])
    if isinstance(houses, list):
        civic_severity = _severity(float(report_counter.get("waste", 0)), 3, 6, 10)
        for house in houses[:2]:
            if not isinstance(house, dict):
                continue
            hotspots.append(
                TwinHotspot(
                    lat=float(house.get("lat", 0.0)),
                    lon=float(house.get("lon", 0.0)),
                    issue="Dense Settlement Stress",
                    severity=civic_severity,
                    source="residential-density",
                )
            )

    return hotspots[:8]


def _nearby_house_count(
    houses: list[dict[str, object]],
    lat: float,
    lon: float,
    radius: float = 0.0028,
) -> float:
    count = 0
    for house in houses:
        if not isinstance(house, dict):
            continue
        h_lat = float(house.get("lat", 0.0))
        h_lon = float(house.get("lon", 0.0))
        if abs(h_lat - lat) <= radius and abs(h_lon - lon) <= radius:
            count += 1
    return float(count)


def _ward_sensor_snapshot(
    indicator: WardIndicator,
    ward_id: int,
    live_weather: dict[str, object] | None = None,
) -> dict[str, float]:
    day_seed = datetime.now(timezone.utc).toordinal()
    rng = np.random.default_rng(ward_id * 43 + day_seed)
    rainfall_anchor: float | None = None
    if isinstance(live_weather, dict):
        rain = live_weather.get("rain_mm")
        precipitation = live_weather.get("precipitation_mm")
        if isinstance(rain, (int, float)) or isinstance(precipitation, (int, float)):
            rainfall_anchor = float(
                max(
                    float(rain) if isinstance(rain, (int, float)) else 0.0,
                    float(precipitation) if isinstance(precipitation, (int, float)) else 0.0,
                )
            )
    if rainfall_anchor is None:
        rainfall = float(np.clip(rng.normal(22 + indicator.flood_risk_index * 84, 10), 0.0, 140.0))
    else:
        rainfall = float(np.clip(rainfall_anchor * 7.2 + rng.normal(8 + indicator.flood_risk_index * 18, 4), 0.0, 140.0))
    pump_runtime = float(np.clip(rng.normal(6.4 - indicator.flood_risk_index * 2.2, 1.4), 0.4, 12.0))
    gps_stagnation = float(np.clip(rng.normal(0.3 + indicator.flood_risk_index * 0.45, 0.12), 0.0, 1.0))
    return {
        "rainfall_sensor_mm": rainfall,
        "pump_runtime_hours": pump_runtime,
        "gps_stagnation": gps_stagnation,
    }


def _drain_report_pressure(report_rows: list[CitizenReport]) -> tuple[float, float]:
    total = max(len(report_rows), 1)
    blocked = sum(1 for row in report_rows if row.category == "blocked_drain")
    flooding = sum(1 for row in report_rows if row.category == "flooding")
    return (
        float(np.clip((blocked + flooding * 0.7) / total, 0.0, 1.0)),
        float(np.clip((flooding + blocked * 0.4) / total, 0.0, 1.0)),
    )


def _build_informal_tile(
    center_lat: float,
    center_lon: float,
    houses: list[dict[str, object]],
    water_midpoints: list[tuple[float, float]],
    rng: np.random.Generator,
    tile_size: int = 64,
) -> np.ndarray:
    tile = rng.normal(0.3, 0.08, size=(tile_size, tile_size)).astype(np.float32)
    radius = 0.0048
    for house in houses:
        if not isinstance(house, dict):
            continue
        lat = float(house.get("lat", 0.0))
        lon = float(house.get("lon", 0.0))
        d_lat = lat - center_lat
        d_lon = lon - center_lon
        if abs(d_lat) > radius or abs(d_lon) > radius:
            continue
        px = int(np.clip(((d_lon / radius) + 1) * 0.5 * (tile_size - 1), 0, tile_size - 1))
        py = int(np.clip(((d_lat / radius) + 1) * 0.5 * (tile_size - 1), 0, tile_size - 1))
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                xx = int(np.clip(px + ox, 0, tile_size - 1))
                yy = int(np.clip(py + oy, 0, tile_size - 1))
                tile[yy, xx] += float(0.16 - 0.02 * (abs(ox) + abs(oy)))

    if water_midpoints:
        nearest = min(
            abs(center_lat - w_lat) + abs(center_lon - w_lon)
            for w_lat, w_lon in water_midpoints
        )
    else:
        nearest = 0.03
    water_influence = float(np.clip(1.0 - nearest / 0.06, 0.0, 1.0))
    tile += water_influence * 0.07
    tile = np.clip(tile, 0.0, 1.0)
    return tile


def _build_blocked_drain_network(
    layers: dict[str, object],
    indicator: WardIndicator,
    ward: Ward,
    report_rows: list[CitizenReport],
    live_weather: dict[str, object] | None = None,
    target_blocked_count: int | None = None,
) -> list[dict[str, object]]:
    drain_paths = layers.get("drains", [])
    water_paths = layers.get("waterbodies", [])
    river_paths = layers.get("rivers", [])
    houses = layers.get("houses", [])
    if not isinstance(drain_paths, list):
        drain_paths = []
    if not isinstance(water_paths, list):
        water_paths = []
    if not isinstance(river_paths, list):
        river_paths = []
    water_paths = [*water_paths, *river_paths]
    if not isinstance(houses, list):
        houses = []

    if not drain_paths:
        return []

    sensor = _ward_sensor_snapshot(indicator, ward.id, live_weather=live_weather)
    citizen_pressure, _ = _drain_report_pressure(report_rows)
    water_midpoints = [_line_midpoint(path) for path in water_paths if isinstance(path, list) and path]
    feature_rows: list[dict[str, object]] = []
    for path_idx, path in enumerate(drain_paths):
        if not isinstance(path, list) or len(path) < 2:
            continue
        for edge_idx in range(len(path) - 1):
            edge = [path[edge_idx], path[edge_idx + 1]]
            if not all(isinstance(pt, list) and len(pt) == 2 for pt in edge):
                continue
            lat, lon = _line_midpoint(edge)
            if water_midpoints:
                min_dist = min(abs(lat - w_lat) + abs(lon - w_lon) for w_lat, w_lon in water_midpoints)
            else:
                min_dist = 0.035
            water_proximity = float(np.clip(1.0 - (min_dist / 0.06), 0.0, 1.0))
            house_density = _nearby_house_count(houses, lat, lon, radius=0.0028)
            segment_length_m = float(
                np.clip(
                    (((edge[1][0] - edge[0][0]) * 111_000) ** 2 + ((edge[1][1] - edge[0][1]) * 111_000) ** 2)
                    ** 0.5,
                    4.0,
                    280.0,
                )
            )
            last_maintenance_days = float(np.clip(42 + (path_idx % 17) * 9 + (edge_idx % 4) * 6, 2.0, 260.0))
            feature_rows.append(
                {
                    "path": edge,
                    "lat": lat,
                    "lon": lon,
                    "segment_length_m": segment_length_m,
                    "water_proximity": water_proximity,
                    "house_density": house_density,
                    "citizen_pressure": float(np.clip(citizen_pressure + (path_idx % 5) * 0.04, 0.0, 1.0)),
                    "rainfall_sensor_mm": sensor["rainfall_sensor_mm"],
                    "pump_runtime_hours": sensor["pump_runtime_hours"],
                    "last_maintenance_days": last_maintenance_days,
                }
            )

    if not feature_rows:
        return []

    frame = pd.DataFrame(
        [
            {
                "segment_length_m": row["segment_length_m"],
                "water_proximity": row["water_proximity"],
                "house_density": row["house_density"],
                "citizen_pressure": row["citizen_pressure"],
                "rainfall_sensor_mm": row["rainfall_sensor_mm"],
                "pump_runtime_hours": row["pump_runtime_hours"],
                "last_maintenance_days": row["last_maintenance_days"],
            }
            for row in feature_rows
        ]
    )
    probabilities = MODEL_HUB.predict_drain_blockage(frame)
    for row, prob in zip(feature_rows, probabilities, strict=True):
        score = float(np.clip(prob, 0.01, 0.99))
        reasons: list[str] = []
        if float(row["water_proximity"]) >= 0.62:
            reasons.append("near waterbody")
        if float(row["citizen_pressure"]) >= 0.45:
            reasons.append("citizen flood/drain alerts")
        if float(row["last_maintenance_days"]) >= 100:
            reasons.append("long since maintenance")
        if not reasons:
            reasons.append("model-estimated accumulation risk")
        row["risk_score"] = score
        row["severity"] = _severity(score, 0.2, 0.4, 0.65)
        row["label"] = f"Blocked segment P={score:.2f} ({', '.join(reasons[:2])})"

    blocked_target = (
        int(target_blocked_count)
        if isinstance(target_blocked_count, int)
        else int(indicator.blocked_drain_count)
    )
    target_count = int(max(0, min(blocked_target, len(feature_rows))))
    if target_count == 0:
        return []
    ranked = sorted(feature_rows, key=lambda x: float(x["risk_score"]), reverse=True)
    return ranked[:target_count]


def _build_blocked_drain_markers(
    blocked_network: list[dict[str, object]],
) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    for idx, segment in enumerate(blocked_network):
        path = segment.get("path")
        if not isinstance(path, list) or len(path) < 2:
            continue
        pt = path[(idx * 7) % len(path)]
        if not isinstance(pt, list) or len(pt) != 2:
            continue
        local_risk = float(segment.get("risk_score", 0.5))
        severity = str(segment.get("severity", _severity(local_risk, 0.2, 0.4, 0.65)))
        markers.append(
            {
                "lat": float(pt[0]),
                "lon": float(pt[1]),
                "severity": severity,
                "risk_score": round(local_risk, 3),
                "label": f"Blocked Drain #{idx + 1} (network)",
            }
        )
    return markers


def _build_flood_zones(
    layers: dict[str, object],
    indicator: WardIndicator,
    ward: Ward,
    blocked_network: list[dict[str, object]],
    report_rows: list[CitizenReport],
    live_weather: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    water_paths = layers.get("waterbodies", [])
    river_paths = layers.get("rivers", [])
    drain_paths = layers.get("drains", [])
    houses = layers.get("houses", [])
    if not isinstance(water_paths, list):
        water_paths = []
    if not isinstance(river_paths, list):
        river_paths = []
    if not isinstance(drain_paths, list):
        drain_paths = []
    if not isinstance(houses, list):
        houses = []
    water_paths = [*water_paths, *river_paths]
    source_paths = water_paths + drain_paths
    if not source_paths:
        return []

    water_mid = [_line_midpoint(path) for path in water_paths if isinstance(path, list) and path]
    drain_mid = [_line_midpoint(path) for path in drain_paths if isinstance(path, list) and path]
    candidates = water_mid + drain_mid
    if not candidates:
        return []
    candidates = candidates[:320]

    sensor = _ward_sensor_snapshot(indicator, ward.id, live_weather=live_weather)
    _, flood_pressure = _drain_report_pressure(report_rows)
    mean_blocked = (
        float(np.mean([float(seg.get("risk_score", 0.0)) for seg in blocked_network]))
        if blocked_network
        else float(np.clip(indicator.blocked_drain_count / max(len(drain_paths), 1), 0.0, 1.0))
    )
    rows: list[dict[str, float]] = []
    for idx, (lat, lon) in enumerate(candidates):
        if water_mid:
            near_water = min(abs(lat - w_lat) + abs(lon - w_lon) for w_lat, w_lon in water_mid)
        else:
            near_water = 0.05
        water_proximity = float(np.clip(1.0 - near_water / 0.07, 0.0, 1.0))
        impervious_surface = float(np.clip(_nearby_house_count(houses, lat, lon, 0.0032) / 34.0, 0.0, 1.0))
        drainage_congestion = float(np.clip(mean_blocked + (idx % 5) * 0.02, 0.0, 1.0))
        elevation_proxy = float(np.clip(0.74 - ((lat + lon) % 0.03) * 9.5, 0.05, 0.95))
        rows.append(
            {
                "water_proximity": water_proximity,
                "drainage_congestion": drainage_congestion,
                "impervious_surface": impervious_surface,
                "elevation_proxy": elevation_proxy,
                "rainfall_sensor_mm": sensor["rainfall_sensor_mm"],
                "citizen_flood_pressure": float(np.clip(flood_pressure + (idx % 4) * 0.05, 0.0, 1.0)),
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    frame = pd.DataFrame(rows)
    risk_pred = MODEL_HUB.predict_flood_risk(frame)
    scored = []
    for row, risk in zip(rows, risk_pred, strict=True):
        score = float(np.clip(risk, 0.03, 0.99))
        scored.append({**row, "risk_score": score})
    scored.sort(key=lambda x: float(x["risk_score"]), reverse=True)

    dynamic_threshold = float(np.quantile([row["risk_score"] for row in scored], 0.72))
    selected = [row for row in scored if float(row["risk_score"]) >= dynamic_threshold]
    if not selected:
        selected = scored[: max(1, min(4, len(scored)))]
    selected = selected[:12]

    exposed_hh = int(indicator.exposed_population / 4.6)
    zones: list[dict[str, object]] = []
    for idx, row in enumerate(selected):
        local_risk = float(row["risk_score"])
        radius = float(round(42 + local_risk * 290 + idx * 6, 1))
        reasons: list[str] = []
        if float(row["water_proximity"]) >= 0.6:
            reasons.append("water proximity")
        if float(row["drainage_congestion"]) >= 0.5:
            reasons.append("drain congestion")
        if float(row["rainfall_sensor_mm"]) >= 60:
            reasons.append("high rainfall pulse")
        if not reasons:
            reasons.append("multi-source model blend")
        zones.append(
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "radius": radius,
                "risk_score": round(local_risk, 3),
                "label": (
                    f"Flood risk P={local_risk:.2f}, ~{max(60, int(exposed_hh / max(len(selected),1)))} HH "
                    f"({', '.join(reasons[:2])})"
                ),
            }
        )
    return zones


def _make_zone_polygon(
    lat: float,
    lon: float,
    lat_radius: float,
    lon_radius: float,
    points: int = 14,
) -> list[list[float]]:
    polygon: list[list[float]] = []
    for idx in range(points):
        angle = 2 * np.pi * (idx / points)
        polygon.append(
            [
                float(round(lat + np.sin(angle) * lat_radius, 6)),
                float(round(lon + np.cos(angle) * lon_radius, 6)),
            ]
        )
    polygon.append(polygon[0])
    return polygon


def _build_informal_zones(
    layers: dict[str, object],
    indicator: WardIndicator,
    ward: Ward,
    boundary: list[list[float]],
) -> list[dict[str, object]]:
    houses = layers.get("houses", [])
    water = layers.get("waterbodies", [])
    rivers = layers.get("rivers", [])
    if not isinstance(houses, list):
        houses = []
    if not isinstance(water, list):
        water = []
    if not isinstance(rivers, list):
        rivers = []
    water = [*water, *rivers]

    zone_count = int(max(1, min(10, round(indicator.informal_area_pct / 10))))
    target_hh = int(ward.households * indicator.informal_area_pct / 100.0)
    if not boundary:
        return []

    lats = [pt[0] for pt in boundary]
    lons = [pt[1] for pt in boundary]
    lat_span = max(max(lats) - min(lats), 0.005)
    lon_span = max(max(lons) - min(lons), 0.005)
    rng = np.random.default_rng(ward.id * 17 + 53)
    water_midpoints = [_line_midpoint(path) for path in water if isinstance(path, list) and path]

    candidate_centers: list[tuple[float, float]] = []
    for idx in range(min(40, max(zone_count * 5, 8))):
        if houses:
            house = houses[(idx * 7) % len(houses)]
            if isinstance(house, dict):
                candidate_centers.append(
                    (
                        float(house.get("lat", lats[0])),
                        float(house.get("lon", lons[0])),
                    )
                )
                continue
        candidate_centers.append(
            (float(rng.uniform(min(lats), max(lats))), float(rng.uniform(min(lons), max(lons))))
        )

    if not candidate_centers:
        return []

    tiles = np.zeros((len(candidate_centers), 1, 64, 64), dtype=np.float32)
    local_density = np.zeros((len(candidate_centers),), dtype=np.float32)
    for idx, (lat, lon) in enumerate(candidate_centers):
        tiles[idx, 0] = _build_informal_tile(lat, lon, houses, water_midpoints, rng, tile_size=64)
        local_density[idx] = float(np.clip(_nearby_house_count(houses, lat, lon, 0.0032) / 28.0, 0.0, 1.0))
    seg_scores = MODEL_HUB.infer_informal_area(tiles)

    candidate_rows: list[dict[str, float]] = []
    base_bias = float(np.clip(indicator.informal_area_pct / 100.0, 0.05, 0.95))
    for idx, (lat, lon) in enumerate(candidate_centers):
        score = float(np.clip(seg_scores[idx] * 0.65 + local_density[idx] * 0.25 + base_bias * 0.1, 0.05, 0.99))
        candidate_rows.append({"lat": lat, "lon": lon, "score": score})
    candidate_rows.sort(key=lambda x: float(x["score"]), reverse=True)
    selected = candidate_rows[:zone_count]
    score_sum = max(sum(float(row["score"]) for row in selected), 1e-6)

    zones: list[dict[str, object]] = []
    for idx, row in enumerate(selected):
        lat = float(row["lat"])
        lon = float(row["lon"])
        density = float(row["score"])
        lat_radius = lat_span * (0.06 + density * 0.12)
        lon_radius = lon_span * (0.06 + density * 0.12)
        households_est = int(max(45, round(target_hh * density / score_sum)))
        zones.append(
            {
                "polygon": _make_zone_polygon(lat, lon, lat_radius, lon_radius),
                "density_score": round(density, 3),
                "households_est": households_est,
            }
        )
    return zones


def _status_from_score(score: float) -> Literal["on_track", "watch", "critical"]:
    if score >= 70:
        return "on_track"
    if score >= 50:
        return "watch"
    return "critical"


def _build_sdg11_card(
    ward: Ward,
    indicator: WardIndicator,
    feature_counts: dict[str, int],
) -> SDG11GovernanceCard:
    green_assets = float(feature_counts.get("green", 0) + feature_counts.get("playground", 0))
    road_assets = float(feature_counts.get("highway", 0))

    target_113 = max(0.0, 100.0 - indicator.informal_area_pct * 1.1)
    target_115 = max(0.0, 100.0 - indicator.flood_risk_index * 150.0)
    target_116 = max(0.0, 100.0 - indicator.blocked_drain_count * 2.4)
    target_117 = min(100.0, 25.0 + green_assets * 2.6)
    target_11b = min(100.0, 30.0 + road_assets * 0.05 + indicator.sdg11_score * 0.5)

    targets = [
        SDG11TargetStatus(
            target="11.3 Inclusive Planning",
            score=round(target_113, 2),
            status=_status_from_score(target_113),
            evidence=f"{indicator.informal_area_pct:.1f}% of the ward consist of unplanned informal settlements.",
        ),
        SDG11TargetStatus(
            target="11.5 Disaster Risk Reduction",
            score=round(target_115, 2),
            status=_status_from_score(target_115),
            evidence=f"Active flood vulnerability level detected at {indicator.flood_risk_index:.2f} score.",
        ),
        SDG11TargetStatus(
            target="11.6 Urban Service Quality",
            score=round(target_116, 2),
            status=_status_from_score(target_116),
            evidence=f"AI detected {indicator.blocked_drain_count} critical blockages in the drainage network.",
        ),
        SDG11TargetStatus(
            target="11.7 Public/Green Space Access",
            score=round(target_117, 2),
            status=_status_from_score(target_117),
            evidence=f"Only {int(green_assets)} public green or recreational spaces were found.",
        ),
        SDG11TargetStatus(
            target="11.b Risk-sensitive Governance",
            score=round(target_11b, 2),
            status=_status_from_score(target_11b),
            evidence=f"Evidence-based policy fused with overall SDG score {indicator.sdg11_score:.1f}.",
        ),
    ]

    priority = "Drainage and service maintenance should be prioritized in the next cycle."
    if indicator.sdg11_score >= 65:
        priority = "Maintain current investments and focus on targeted equity interventions."

    return SDG11GovernanceCard(
        ward_id=ward.id,
        ward_name=ward.name,
        overall_score=round(indicator.sdg11_score, 2),
        targets=targets,
        priority_message=priority,
        generated_at=datetime.now(timezone.utc),
    )


def _supporting_agencies(lead: str) -> list[str]:
    base = {
        "City Corporation": ["WASA", "RAJUK"],
        "LGED": ["City Corporation", "RAJUK"],
        "WASA": ["City Corporation", "LGED"],
        "RAJUK": ["City Corporation", "LGED"],
    }
    return base.get(lead, ["City Corporation"])


def _build_interagency_packet(
    ward: Ward,
    ranked_items: list[TopWorkItem],
) -> InterAgencyPacket:
    tasks: list[InterAgencyTask] = []
    for item in ranked_items[:8]:
        timeline_weeks = int(
            max(2, min(14, round(item.estimated_cost_lakh * 2.8 + (3 if item.permit_required else 0))))
        )
        dependency = "Permit clearance before procurement" if item.permit_required else "Direct procurement"
        tasks.append(
            InterAgencyTask(
                intervention_id=item.id,
                action_title=item.title,
                category=item.category,
                lead_agency=item.agency,
                supporting_agencies=_supporting_agencies(item.agency),
                permit_required=item.permit_required,
                estimated_cost_lakh=item.estimated_cost_lakh,
                timeline_weeks=timeline_weeks,
                dependency=dependency,
            )
        )

    checklist = [
        "Validate ward evidence and field verification records.",
        "Approve Top-N prioritization list in municipal planning meeting.",
        "Assign lead and supporting agencies for each micro-work.",
        "Complete permit and procurement checklist.",
        "Issue work order and track milestone updates in audit trail.",
    ]
    return InterAgencyPacket(
        ward_id=ward.id,
        ward_name=ward.name,
        generated_at=datetime.now(timezone.utc),
        checklist=checklist,
        tasks=tasks,
    )


def _civic_priority_adjustments(report_rows: list[CitizenReport]) -> dict[str, float]:
    if not report_rows:
        return {}
    weighted = Counter()
    for row in report_rows:
        weighted[row.category] += float(row.priority_weight)

    normalize = max(sum(weighted.values()), 1.0)
    complaint_ratio = {k: v / normalize for k, v in weighted.items()}
    category_to_work = {
        "blocked_drain": ["Drainage", "Water"],
        "flooding": ["Drainage", "Water"],
        "waste": ["Waste"],
        "road_damage": ["Road"],
        "water_supply": ["Water"],
    }
    boosts: dict[str, float] = {}
    for report_cat, ratio in complaint_ratio.items():
        for work_cat in category_to_work.get(report_cat, []):
            boosts[work_cat] = boosts.get(work_cat, 0.0) + ratio * 0.18
    return boosts


def _notification_message(action: str, details: dict[str, object]) -> tuple[str, str]:
    if action == "live_source_ingested":
        source = str(details.get("source", "source"))
        records = int(details.get("records_ingested", 0))
        return "info", f"Live source {source} ingested {records} new records."
    if action == "live_model_rerun":
        top_changes = int(details.get("top_changes_count", 0))
        return "success", f"Models reran with latest evidence; {top_changes} top-worklist changes detected."
    if action == "odk_submission_ingested":
        issue = str(details.get("category", "issue")).replace("_", " ")
        return "warning", f"ODK/KoBo field report received ({issue})."
    if action == "civic_report_created":
        issue = str(details.get("category", "issue")).replace("_", " ")
        return "warning", f"Citizen complaint logged ({issue})."
    if action == "scenario_simulated":
        budget = float(details.get("budget_lakh", 0.0))
        return "info", f"Scenario rerun completed for budget {budget:.1f} lakh."
    return "info", action.replace("_", " ")


def _to_notification_item(event: object) -> NotificationFeedItem:
    details: dict[str, object]
    try:
        details = json.loads(getattr(event, "details_json", "{}"))
    except Exception:
        details = {}
    severity, message = _notification_message(str(getattr(event, "action", "event")), details)
    source = str(details.get("source", getattr(event, "action", "system")))
    return NotificationFeedItem(
        id=int(getattr(event, "id")),
        timestamp=getattr(event, "timestamp"),
        source=source,
        severity=severity,
        message=message,
        ward_id=getattr(event, "ward_id"),
    )


def _seed_live_text(category: str, idx: int) -> str:
    templates = {
        "blocked_drain": [
            "Drain choke observed near lane {lane}; overflow into roads.",
            "ড্রেন বন্ধ, পানি নামছে না, দ্রুত পরিষ্কার দরকার",
        ],
        "flooding": [
            "Waterlogging persisted after rain in block {lane}.",
            "বৃষ্টির পর রাস্তা ডুবে থাকে, চলাচল বন্ধ",
        ],
        "waste": [
            "Open waste pile found beside canal at sector {lane}.",
            "ডাস্টবিন না থাকায় এখানে ময়লা জমে আছে",
        ],
        "road_damage": [
            "Road patch failed and potholes reopened at junction {lane}.",
            "রাস্তায় ভাঙন, ছোট যানবাহন ঝুঁকিতে",
        ],
        "water_supply": [
            "Low pressure complaint from cluster {lane}.",
            "পানির লাইন দুর্বল, সকালে পানি পাওয়া যাচ্ছে না",
        ],
    }
    pool = templates.get(category, ["Civic report from cluster {lane}."])
    line = pool[idx % len(pool)]
    return line.format(lane=1 + (idx % 24))


@router.get("/wards", response_model=list[WardSummary])
def list_wards(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[WardSummary]:
    rows = db.execute(
        select(Ward, WardIndicator.sdg11_score)
        .join(WardIndicator, WardIndicator.ward_id == Ward.id)
        .order_by(Ward.id.asc())
    ).all()
    return [
        WardSummary(
            id=ward.id,
            code=ward.code,
            name=ward.name,
            area_km2=ward.area_km2,
            population=ward.population,
            households=ward.households,
            last_updated=ward.last_updated,
            sdg11_score=float(sdg),
        )
        for ward, sdg in rows
    ]


@router.get("/ai-components", response_model=list[AIComponentStatus])
def ai_components(_: User = Depends(get_current_user)) -> list[AIComponentStatus]:
    return [
        AIComponentStatus(
            name="Prioritization & Decision Ranker",
            technology="Gradient boosting ranker (LightGBM/XGBoost-style baseline)",
            policy_output="Ordered Top-N worklist with justification and cost-benefit context.",
            status="implemented",
            endpoint="/api/wards/{ward_id}/top-worklist",
        ),
        AIComponentStatus(
            name="Evidence Extractor",
            technology=(
                "Mini U-Net segmentation + AI drainage monitor + local flood-risk model "
                "+ OpenStreetMap/Overpass geospatial feed"
            ),
            policy_output="Model-estimated informal zones, blocked-drain network, and exposure indicators.",
            status="implemented",
            endpoint="/api/wards/{ward_id}/digital-twin-scene",
        ),
        AIComponentStatus(
            name="Policy Scenario Simulator",
            technology=(
                "Real-world constrained prescriptive optimizer "
                "(budget reserve, OPEX cap, permit-share cap, agency capacity, timeline checks)"
            ),
            policy_output="Budget comparison tables and memo-ready tradeoff outputs.",
            status="implemented",
            endpoint="/api/wards/{ward_id}/scenario",
        ),
        AIComponentStatus(
            name="Civic Intelligence",
            technology=(
                "Bangla/English NLP classifier + ODK/KoBo form ingestion + notification routing "
                "+ Open-Meteo live weather feed"
            ),
            policy_output="Citizen/field validation flags and community-priority signal to ranking.",
            status="implemented",
            endpoint="/api/reports/odk-submit",
        ),
        AIComponentStatus(
            name="Governance Summarizer",
            technology="Template-based policy memo + packet generation",
            policy_output="SDG card, inter-agency packet, procurement-oriented PDF/CSV exports.",
            status="implemented",
            endpoint="/api/wards/{ward_id}/sdg11-card",
        ),
    ]


@router.get("/model-trust", response_model=dict[str, object])
def model_trust(_: User = Depends(get_current_user)) -> dict[str, object]:
    return MODEL_HUB.get_trust_summary()


@router.get("/model-cards", response_model=dict[str, object])
def model_cards(_: User = Depends(get_current_user)) -> dict[str, object]:
    return MODEL_HUB.model_cards if MODEL_HUB.model_cards else {}


@router.get("/audit-trail", response_model=list[AuditTrailItem])
def audit_trail(
    ward_id: int | None = Query(default=None),
    limit: int = Query(default=40, ge=5, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[AuditTrailItem]:
    rows = list_audit_events(db=db, ward_id=ward_id, limit=limit)
    items: list[AuditTrailItem] = []
    for row in rows:
        try:
            details = json.loads(row.details_json)
        except Exception:
            details = {}
        items.append(
            AuditTrailItem(
                id=row.id,
                timestamp=row.timestamp,
                actor_username=row.actor_username,
                actor_role=row.actor_role,
                action=row.action,
                ward_id=row.ward_id,
                details=details,
            )
        )
    return items


@router.get("/wards/{ward_id}/notification-feed", response_model=list[NotificationFeedItem])
def ward_notification_feed(
    ward_id: int,
    limit: int = Query(default=20, ge=5, le=120),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[NotificationFeedItem]:
    rows = list_audit_events(db=db, ward_id=ward_id, limit=max(limit * 3, 30))
    interesting_actions = {
        "live_source_ingested",
        "live_model_rerun",
        "odk_submission_ingested",
        "civic_report_created",
        "scenario_simulated",
    }
    filtered = [row for row in rows if row.action in interesting_actions][:limit]
    return [_to_notification_item(row) for row in filtered]


@router.post("/wards/{ward_id}/live-ingest", response_model=LiveUpdateResponse)
def simulate_live_ingest(
    ward_id: int,
    payload: LiveIngestRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("planner", "enumerator")),
) -> LiveUpdateResponse:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")

    before_rank = db.scalars(
        select(Intervention)
        .where(Intervention.ward_id == ward_id)
        .order_by(Intervention.ranking_score.desc())
        .limit(8)
    ).all()
    before_ids = [item.id for item in before_rank]
    boundaries = _load_json_file("ward_boundaries.json")
    boundary = boundaries.get(str(ward_id), json.loads(ward.bbox_json))
    feature_counts = _load_json_file("ward_feature_counts.json").get(str(ward_id), {})
    if not isinstance(feature_counts, dict):
        feature_counts = {}
    map_layers = _load_json_file("ward_map_layers.json")
    layer_payload = map_layers.get(str(ward_id), {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}
    morphology = summarize_ward_morphology(boundary, layer_payload, feature_counts)
    centroid_lat, centroid_lon = _boundary_centroid(boundary)
    live_weather = fetch_open_meteo_current(centroid_lat, centroid_lon)
    live_rainfall_mm = _weather_rainfall_mm(live_weather)
    rainfall_for_calc = live_rainfall_mm if live_rainfall_mm is not None else 0.0
    weather_pressure = float(np.clip(rainfall_for_calc / 68.0, 0.0, 0.14))

    categories = ["blocked_drain", "flooding", "waste", "road_damage", "water_supply"]
    rng_seed = (
        int(datetime.now(timezone.utc).timestamp())
        + ward_id * 41
        + payload.citizen_reports * 7
        + payload.odk_forms * 13
        + payload.sensor_pulses * 17
    )
    rng = np.random.default_rng(rng_seed)
    category_counter = Counter()
    source_events: list[LiveSourceEvent] = []

    for idx in range(payload.citizen_reports):
        category = categories[int(rng.integers(0, len(categories)))]
        text = _seed_live_text(category, idx)
        language = "bangla" if any("\u0980" <= ch <= "\u09ff" for ch in text) else "english"
        sentiment = float(np.clip(rng.normal(-0.62, 0.18), -1.0, 0.2))
        priority = float(
            np.clip(
                0.46 + (0.26 if category in {"blocked_drain", "flooding"} else 0.1) + rng.normal(0, 0.08),
                0.1,
                1.0,
            )
        )
        db.add(
            CitizenReport(
                ward_id=ward_id,
                text=text,
                language=language,
                category=category,
                sentiment_score=sentiment,
                priority_weight=priority,
            )
        )
        category_counter[category] += 1
    if payload.citizen_reports > 0:
        source_events.append(
            LiveSourceEvent(
                source="citizen_portal",
                records_ingested=payload.citizen_reports,
                note="Participatory portal complaints fused into civic-intelligence stream.",
            )
        )
        log_audit_event(
            db,
            action="live_source_ingested",
            user=user,
            ward_id=ward_id,
            details={
                "source": "citizen_portal",
                "records_ingested": payload.citizen_reports,
            },
        )

    for idx in range(payload.odk_forms):
        category = categories[int(rng.choice([0, 1, 2, 2, 3, 4]))]
        text = f"[ODK] {_seed_live_text(category, idx + 31)}"
        language = "bangla" if any("\u0980" <= ch <= "\u09ff" for ch in text) else "english"
        sentiment = float(np.clip(rng.normal(-0.55, 0.2), -1.0, 0.3))
        priority = float(
            np.clip(
                0.52 + (0.24 if category in {"blocked_drain", "flooding"} else 0.12) + rng.normal(0, 0.07),
                0.1,
                1.0,
            )
        )
        db.add(
            CitizenReport(
                ward_id=ward_id,
                text=text,
                language=language,
                category=category,
                sentiment_score=sentiment,
                priority_weight=priority,
            )
        )
        category_counter[category] += 1
    if payload.odk_forms > 0:
        source_events.append(
            LiveSourceEvent(
                source="odk_kobo",
                records_ingested=payload.odk_forms,
                note="Enumerator mobile forms synchronized from field submissions.",
            )
        )
        log_audit_event(
            db,
            action="live_source_ingested",
            user=user,
            ward_id=ward_id,
            details={
                "source": "odk_kobo",
                "records_ingested": payload.odk_forms,
            },
        )

    blocked_signals = int(category_counter.get("blocked_drain", 0) + category_counter.get("flooding", 0))
    network_scale = max(estimate_blocked_network_scale(morphology), 1.0)
    rainfall_pressure = float(np.clip(rainfall_for_calc / 95.0, 0.0, 1.0))
    sensor_pressure = float(np.clip(payload.sensor_pulses / 16.0, 0.0, 1.0))
    complaint_pressure = float(np.clip(blocked_signals / 18.0, 0.0, 1.0))
    blocked_delta = int(
        np.clip(
            round(
                max(2.0, network_scale * 0.09)
                * (
                    0.44 * rainfall_pressure
                    + 0.34 * complaint_pressure
                    + 0.22 * sensor_pressure
                )
            ),
            -4,
            20,
        )
    )
    indicator.blocked_drain_count = int(
        np.clip(indicator.blocked_drain_count + blocked_delta, 0, int(round(network_scale)))
    )
    indicator.flood_risk_index = score_live_flood_risk(
        baseline_flood_risk=float(indicator.flood_risk_index),
        blocked_drain_count=int(indicator.blocked_drain_count),
        green_deficit_index=float(indicator.green_deficit_index),
        informal_area_pct=float(indicator.informal_area_pct),
        morphology=morphology,
        rainfall_mm=live_rainfall_mm,
        blocked_signals=blocked_signals,
        sensor_pulses=payload.sensor_pulses,
    )
    indicator.exposed_population = estimate_exposed_population(
        ward.population,
        float(indicator.flood_risk_index),
        int(indicator.blocked_drain_count),
        float(indicator.informal_area_pct),
        morphology,
    )
    indicator.sdg11_score = score_sdg11(
        flood_risk_index=float(indicator.flood_risk_index),
        green_deficit_index=float(indicator.green_deficit_index),
        blocked_drain_count=int(indicator.blocked_drain_count),
        informal_area_pct=float(indicator.informal_area_pct),
        morphology=morphology,
    )
    ward.last_updated = datetime.now(timezone.utc)
    db.commit()

    if payload.sensor_pulses > 0:
        source_events.append(
            LiveSourceEvent(
                source="sensor_feed",
                records_ingested=payload.sensor_pulses,
                note=(
                    "Rain/flood sensor pulse changed indicators "
                    f"(rainfall={rainfall_for_calc:.1f}mm, network_scale={network_scale:.1f})."
                ),
            )
        )
        log_audit_event(
            db,
            action="live_source_ingested",
            user=user,
            ward_id=ward_id,
            details={
                "source": "sensor_feed",
                "records_ingested": payload.sensor_pulses,
                "blocked_delta": blocked_delta,
                "weather_pressure": round(weather_pressure, 4),
            },
        )

    if str(live_weather.get("status", "unavailable")) != "unavailable":
        rainfall_note = (
            f"{rainfall_for_calc:.1f}mm"
            if live_rainfall_mm is not None
            else "no-precip-data"
        )
        source_events.append(
            LiveSourceEvent(
                source="open_meteo",
                records_ingested=1,
                note=f"Open-Meteo rainfall/temperature feed ingested ({rainfall_note} current precipitation).",
            )
        )
        log_audit_event(
            db,
            action="live_source_ingested",
            user=user,
            ward_id=ward_id,
            details={
                "source": "open_meteo",
                "records_ingested": 1,
                "rainfall_mm": (
                    round(live_rainfall_mm, 3)
                    if live_rainfall_mm is not None
                    else None
                ),
                "weather_status": str(live_weather.get("status", "unknown")),
            },
        )

    reranked = top_worklist(
        ward_id=ward_id,
        top_n=10,
        db=db,
        _=user,
        _emit_audit=False,
    )
    top_changes = [item.title for item in reranked.items[:8] if item.id not in before_ids][:5]
    if not top_changes:
        top_changes = ["No project replacement in Top-8; scores still recalibrated with fresh evidence."]
    source_events.append(
        LiveSourceEvent(
            source="model_rerun",
            records_ingested=len(reranked.items),
            note="Prioritization and scenario models reran after data fusion.",
        )
    )
    log_audit_event(
        db,
        action="live_model_rerun",
        user=user,
        ward_id=ward_id,
        details={
            "top_changes_count": len([title for title in top_changes if not title.startswith("No project")]),
            "blocked_drain_count": indicator.blocked_drain_count,
            "flood_risk_index": round(indicator.flood_risk_index, 4),
        },
    )

    return LiveUpdateResponse(
        ward_id=ward_id,
        triggered_at=datetime.now(timezone.utc),
        sources=source_events,
        updated_indicators=WardIndicatorResponse(
            informal_area_pct=indicator.informal_area_pct,
            blocked_drain_count=indicator.blocked_drain_count,
            green_deficit_index=indicator.green_deficit_index,
            flood_risk_index=indicator.flood_risk_index,
            sdg11_score=indicator.sdg11_score,
            exposed_population=indicator.exposed_population,
        ),
        live_weather=LiveWeatherSnapshot(
            source=str(live_weather.get("source", "open-meteo")),
            status=str(live_weather.get("status", "unknown")),
            observed_at=live_weather.get("observed_at")
            if isinstance(live_weather.get("observed_at"), datetime)
            else None,
            rainfall_mm=(
                round(live_rainfall_mm, 3)
                if live_rainfall_mm is not None
                else None
            ),
            temperature_c=(
                float(live_weather["temperature_c"])
                if isinstance(live_weather.get("temperature_c"), (int, float))
                else None
            ),
            wind_speed_kmh=(
                float(live_weather["wind_speed_kmh"])
                if isinstance(live_weather.get("wind_speed_kmh"), (int, float))
                else None
            ),
        ),
        top_changes=top_changes,
    )


@router.get("/wards/{ward_id}/digital-twin", response_model=WardDigitalTwin)
def get_digital_twin(
    ward_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> WardDigitalTwin:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")

    boundaries = _load_json_file("ward_boundaries.json")
    feature_counts = _load_json_file("ward_feature_counts.json")
    map_layers = _load_json_file("ward_map_layers.json")
    layer_payload = map_layers.get(str(ward_id), {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}
    _, effective_blocked_count, integrity_notes = _resolve_drain_geometry(layer_payload, indicator)

    return WardDigitalTwin(
        ward=WardSummary(
            id=ward.id,
            code=ward.code,
            name=ward.name,
            area_km2=ward.area_km2,
            population=ward.population,
            households=ward.households,
            last_updated=ward.last_updated,
            sdg11_score=indicator.sdg11_score,
        ),
        indicators=WardIndicatorResponse(
            informal_area_pct=indicator.informal_area_pct,
            blocked_drain_count=effective_blocked_count,
            green_deficit_index=indicator.green_deficit_index,
            flood_risk_index=indicator.flood_risk_index,
            sdg11_score=indicator.sdg11_score,
            exposed_population=indicator.exposed_population,
        ),
        boundary=boundaries.get(str(ward_id), json.loads(ward.bbox_json)),
        feature_counts=feature_counts.get(str(ward_id), {}),
        integrity_notes=integrity_notes,
    )


@router.get("/wards/{ward_id}/sdg11-card", response_model=SDG11GovernanceCard)
def ward_sdg11_card(
    ward_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SDG11GovernanceCard:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")
    feature_counts = _load_json_file("ward_feature_counts.json").get(str(ward_id), {})
    if not isinstance(feature_counts, dict):
        feature_counts = {}

    card = _build_sdg11_card(ward, indicator, feature_counts)
    log_audit_event(
        db,
        action="sdg11_card_generated",
        user=current_user,
        ward_id=ward_id,
        details={"overall_score": card.overall_score},
    )
    return card


@router.get("/wards/{ward_id}/digital-twin-scene", response_model=WardTwinSceneResponse)
def get_digital_twin_scene(
    ward_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> WardTwinSceneResponse:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")

    boundaries = _load_json_file("ward_boundaries.json")
    feature_counts = _load_json_file("ward_feature_counts.json")
    map_layers = _load_json_file("ward_map_layers.json")
    dataset_meta = _load_dataset_meta()
    layer_payload = map_layers.get(str(ward_id), {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}

    roads = layer_payload.get("roads", [])
    drains, effective_blocked_count, integrity_notes = _resolve_drain_geometry(layer_payload, indicator)
    rivers = layer_payload.get("rivers", [])
    waterbodies = layer_payload.get("waterbodies", [])
    houses = layer_payload.get("houses", [])
    playgrounds = layer_payload.get("playgrounds", [])
    parks = layer_payload.get("parks", [])
    summary = layer_payload.get("summary", {})
    boundary = boundaries.get(str(ward_id), json.loads(ward.bbox_json))
    centroid_lat, centroid_lon = _boundary_centroid(boundary)
    live_weather = fetch_open_meteo_current(centroid_lat, centroid_lon)

    report_rows = db.scalars(
        select(CitizenReport)
        .where(CitizenReport.ward_id == ward_id)
        .order_by(CitizenReport.created_at.desc())
        .limit(60)
    ).all()
    blocked_drain_network = _build_blocked_drain_network(
        {**layer_payload, "drains": drains},
        indicator,
        ward,
        report_rows,
        live_weather=live_weather,
        target_blocked_count=effective_blocked_count,
    )
    blocked_drains = _build_blocked_drain_markers(blocked_drain_network)
    flood_zones = _build_flood_zones(
        {**layer_payload, "drains": drains},
        indicator,
        ward,
        blocked_drain_network,
        report_rows,
        live_weather=live_weather,
    )
    informal_zones = _build_informal_zones(layer_payload, indicator, ward, boundary)
    interventions = db.scalars(select(Intervention).where(Intervention.ward_id == ward_id)).all()

    merged_summary, summary_notes = _rendered_layer_summary(
        source_summary=summary if isinstance(summary, dict) else None,
        roads=roads,
        drains=drains,
        rivers=rivers,
        waterbodies=waterbodies,
        houses=houses,
        playgrounds=playgrounds,
        parks=parks,
        blocked_drain_network=blocked_drain_network,
        blocked_drains=blocked_drains,
        flood_zones=flood_zones,
        informal_zones=informal_zones,
    )
    for note in summary_notes:
        if note not in integrity_notes:
            integrity_notes.append(note)
    data_sources = _public_source_payload(
        dataset_meta=dataset_meta,
        weather=live_weather,
        river_segments=int(len(rivers) if isinstance(rivers, list) else 0),
    )

    layers = TwinLayers(
        roads=roads,
        drains=drains,
        rivers=rivers if isinstance(rivers, list) else [],
        blocked_drain_network=[TwinBlockedDrainSegment(**segment) for segment in blocked_drain_network],
        waterbodies=waterbodies,
        houses=[TwinHouse(**house) for house in houses if isinstance(house, dict)],
        playgrounds=[
            TwinAreaAsset(**asset) for asset in playgrounds if isinstance(asset, dict)
        ],
        parks=[TwinAreaAsset(**asset) for asset in parks if isinstance(asset, dict)],
        blocked_drains=blocked_drains,
        flood_zones=flood_zones,
        informal_zones=informal_zones,
        summary=merged_summary,
    )

    problems = _build_problems(indicator, report_rows, blocked_drain_count=effective_blocked_count)
    hotspots = _build_hotspots(
        {**layer_payload, "drains": drains},
        indicator,
        report_rows,
        blocked_drain_count=effective_blocked_count,
    )
    actions_taken = _build_actions(interventions)

    return WardTwinSceneResponse(
        ward=WardSummary(
            id=ward.id,
            code=ward.code,
            name=ward.name,
            area_km2=ward.area_km2,
            population=ward.population,
            households=ward.households,
            last_updated=ward.last_updated,
            sdg11_score=indicator.sdg11_score,
        ),
        boundary=boundary,
        layers=layers,
        feature_counts=feature_counts.get(str(ward_id), {}),
        scores=WardIndicatorResponse(
            informal_area_pct=indicator.informal_area_pct,
            blocked_drain_count=effective_blocked_count,
            green_deficit_index=indicator.green_deficit_index,
            flood_risk_index=indicator.flood_risk_index,
            sdg11_score=indicator.sdg11_score,
            exposed_population=indicator.exposed_population,
        ),
        data_sources=data_sources,
        problems=problems,
        actions_taken=actions_taken,
        hotspots=hotspots,
        integrity_notes=integrity_notes,
    )


@router.get("/wards/{ward_id}/workflow", response_model=WardWorkflowResponse)
def ward_workflow(
    ward_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WardWorkflowResponse:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")

    dataset_meta = _load_dataset_meta()
    boundaries = _load_json_file("ward_boundaries.json")
    boundary = boundaries.get(str(ward_id), json.loads(ward.bbox_json))
    centroid_lat, centroid_lon = _boundary_centroid(boundary)
    live_weather = fetch_open_meteo_current(centroid_lat, centroid_lon)

    map_layers = _load_json_file("ward_map_layers.json")
    layer_payload = map_layers.get(str(ward_id), {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}
    layer_summary_raw = layer_payload.get("summary", {})
    if not isinstance(layer_summary_raw, dict):
        layer_summary_raw = {}
    layer_summary = {str(key): int(value) for key, value in layer_summary_raw.items()}

    report_rows = db.scalars(
        select(CitizenReport)
        .where(CitizenReport.ward_id == ward_id)
        .order_by(CitizenReport.created_at.desc())
        .limit(160)
    ).all()
    interventions = db.scalars(select(Intervention).where(Intervention.ward_id == ward_id)).all()

    source_inventory = _workflow_source_inventory(
        ward=ward,
        dataset_meta=dataset_meta,
        layer_summary=layer_summary,
        interventions=interventions,
        report_rows=report_rows,
        live_weather=live_weather,
    )
    top_list = top_worklist(
        ward_id=ward_id,
        top_n=10,
        db=db,
        _=current_user,
        _emit_audit=False,
    )
    candidate_micro_works = _workflow_candidate_proposals(
        ward=ward,
        indicator=indicator,
        layer_summary=layer_summary,
        report_rows=report_rows,
        ranked_items=top_list.items,
    )
    item_dicts = [item.model_dump() for item in top_list.items]
    scenario_payload = simulate_policy_scenario(item_dicts, 9.0) if item_dicts else {
        "selected_projects": [],
        "used_budget_lakh": 0.0,
        "remaining_budget_lakh": 9.0,
        "impacted_households": 0,
        "estimated_sdg11_gain": 0.0,
    }
    packet = _build_interagency_packet(ward=ward, ranked_items=top_list.items)

    complaint_counter = Counter(row.category for row in report_rows)
    recent_audit_rows = list_audit_events(db=db, ward_id=ward_id, limit=40)
    validation_events = len(
        [
            row
            for row in recent_audit_rows
            if row.action
            in {
                "civic_report_created",
                "odk_submission_ingested",
                "live_source_ingested",
                "live_model_rerun",
            }
        ]
    )
    recent_odk = sum(1 for row in report_rows if row.text.startswith("[ODK"))
    segmentation_train_tiles, ward_tiles = _segmentation_inventory()
    vector_layers = sum(
        1
        for key in ["roads", "drains", "rivers", "waterbodies", "houses", "playgrounds", "parks"]
        if int(layer_summary.get(key, 0)) > 0
    )
    mapped_candidates = sum(1 for item in candidate_micro_works if item.mapped_intervention_id is not None)

    stages = [
        WorkflowStageStatus(
            id="data_intake",
            title="Data Intake",
            status="implemented",
            summary="The ward ingests public map layers, population baseline, intervention register, and citizen/field reports.",
            metrics={
                "sources": len(source_inventory),
                "citizen_reports": len(report_rows),
                "asset_records": int(
                    sum(int(layer_summary.get(key, 0)) for key in ["houses", "playgrounds", "parks", "roads", "drains"])
                ),
                "intervention_records": len(interventions),
            },
            details=[
                f"Boundary source: {dataset_meta.get('boundary_source', 'unknown')}",
                f"Map source: {dataset_meta.get('map_source', 'unknown')}",
                "Citizen portal, ODK/KoBo, and live weather inputs are connected to the ward pipeline.",
            ],
        ),
        WorkflowStageStatus(
            id="standardize",
            title="Preprocess & Standardize",
            status="implemented" if vector_layers >= 4 and ward_tiles > 0 else "partial",
            summary="Imagery tiles, ward vectors, and civic reports are normalized into aligned ward-scale inputs with provenance tags.",
            metrics={
                "vector_layers": vector_layers,
                "ward_tiles": ward_tiles,
                "training_tiles": segmentation_train_tiles,
                "normalized_reports": len(report_rows),
            },
            details=[
                f"Dataset version: {dataset_meta.get('dataset_version', 'unknown')}",
                "Boundary, road, drain, river, waterbody, and asset layers are aligned to the same ward geometry.",
                "Citizen and field reports are normalized into category, sentiment, and priority-weight features.",
            ],
        ),
        WorkflowStageStatus(
            id="evidence_extraction",
            title="Evidence Extraction",
            status="implemented",
            summary="AI models convert imagery, vectors, weather, and citizen signals into ward indicators and hotspot layers.",
            metrics={
                "informal_area_pct": round(float(indicator.informal_area_pct), 2),
                "blocked_drain_count": int(indicator.blocked_drain_count),
                "flood_risk_index": round(float(indicator.flood_risk_index), 4),
                "exposed_population": int(indicator.exposed_population),
            },
            details=[
                "Mini U-Net drives informal-area inference from ward imagery tiles.",
                "Drain blockage and flood-risk models generate hotspot evidence for the digital twin.",
                "Evidence outputs are exposed directly in the map, indicators, and workflow summaries.",
            ],
        ),
        WorkflowStageStatus(
            id="candidate_generation",
            title="Candidate Generation",
            status="implemented" if candidate_micro_works else "partial",
            summary="The system auto-proposes feasible micro-works from current evidence and maps them to the intervention library used by the ranker.",
            metrics={
                "generated_candidates": len(candidate_micro_works),
                "mapped_to_ranker": mapped_candidates,
                "agencies_covered": len({item.agency for item in candidate_micro_works}),
                "permit_candidates": sum(1 for item in candidate_micro_works if item.permit_required),
            },
            details=[
                "Candidate proposals are built from blocked-drain, flood, green-deficit, road-access, and complaint-pressure signals.",
                "Each proposal includes jurisdiction, permit flag, rough unit cost, and expected beneficiaries.",
                "Candidates are mapped to the ranked intervention library so the workflow remains continuous into policy ranking.",
            ],
        ),
        WorkflowStageStatus(
            id="rank_simulate",
            title="Rank & Simulate",
            status="implemented" if top_list.items else "partial",
            summary="Ranked Top-N worklists and constrained budget simulations are computed from the current candidate pool.",
            metrics={
                "top_worklist_items": len(top_list.items),
                "selected_projects": len(scenario_payload["selected_projects"]),
                "used_budget_lakh": round(float(scenario_payload["used_budget_lakh"]), 3),
                "estimated_sdg11_gain": round(float(scenario_payload["estimated_sdg11_gain"]), 3),
            },
            details=[
                "The ranker scores interventions using cost-efficiency, equity, urgency, and feasibility.",
                "The simulator applies reserve, OPEX, permit-share, agency-capacity, and timeline constraints.",
                "Selected-project reasoning is exposed for transparent municipal decision making.",
            ],
        ),
        WorkflowStageStatus(
            id="produce_policy_packets",
            title="Produce Policy Packets",
            status="implemented" if packet.tasks else "partial",
            summary="The system generates Top-N worklists, inter-agency packets, and exportable memo formats for operational follow-through.",
            metrics={
                "packet_tasks": len(packet.tasks),
                "checklist_items": len(packet.checklist),
                "export_formats": 2,
                "memo_budget_reference_lakh": 9.0,
            },
            details=[
                "CSV export supports budget and worklist review.",
                "PDF policy memo packages scenario outputs for finance officers.",
                "Inter-agency packet assigns lead/supporting agencies and implementation dependencies.",
            ],
        ),
        WorkflowStageStatus(
            id="field_civic_validation",
            title="Field & Civic Validation",
            status="implemented" if report_rows else "partial",
            summary="Citizen complaints, ODK/KoBo field reports, and live-ingest events feed back into ranking and audit trails.",
            metrics={
                "recent_reports": len(report_rows),
                "odk_like_reports": recent_odk,
                "validation_events": validation_events,
                "top_complaint_category": complaint_counter.most_common(1)[0][0] if complaint_counter else "none",
            },
            details=[
                "Bangla/English NLP classification tags new complaints before they enter the decision pipeline.",
                "ODK/KoBo field submissions are stored and reflected in notifications and audit logs.",
                "Live-ingest reruns ranking after new reports and sensor/weather evidence arrive.",
            ],
        ),
    ]

    return WardWorkflowResponse(
        ward=WardSummary(
            id=ward.id,
            code=ward.code,
            name=ward.name,
            area_km2=ward.area_km2,
            population=ward.population,
            households=ward.households,
            last_updated=ward.last_updated,
            sdg11_score=indicator.sdg11_score,
        ),
        generated_at=datetime.now(timezone.utc),
        workflow_complete=all(stage.status == "implemented" for stage in stages),
        stages=stages,
        source_inventory=source_inventory,
        candidate_micro_works=candidate_micro_works,
    )


@router.get("/wards/{ward_id}/interagency-packet", response_model=InterAgencyPacket)
def ward_interagency_packet(
    ward_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InterAgencyPacket:
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")

    ranked = top_worklist(
        ward_id=ward_id,
        top_n=12,
        db=db,
        _=current_user,
        _emit_audit=False,
    )
    packet = _build_interagency_packet(ward=ward, ranked_items=ranked.items)
    log_audit_event(
        db,
        action="interagency_packet_generated",
        user=current_user,
        ward_id=ward_id,
        details={"task_count": len(packet.tasks)},
    )
    return packet


@router.get("/wards/{ward_id}/top-worklist", response_model=TopWorklistResponse)
def top_worklist(
    ward_id: int,
    top_n: int = Query(default=10, ge=3, le=30),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
    _emit_audit: bool = True,
) -> TopWorklistResponse:
    interventions = db.scalars(
        select(Intervention).where(Intervention.ward_id == ward_id)
    ).all()
    if not interventions:
        raise HTTPException(status_code=404, detail="No interventions found for ward.")

    frame = pd.DataFrame(
        [
            {
                "id": row.id,
                "category": row.category,
                "agency": row.agency,
                "permit_required": row.permit_required,
                "estimated_cost_lakh": row.estimated_cost_lakh,
                "expected_beneficiaries": row.expected_beneficiaries,
                "feasibility": row.feasibility,
                "equity_need": row.equity_need,
                "urgency": row.urgency,
            }
            for row in interventions
        ]
    )

    scores = MODEL_HUB.score_interventions(frame)
    report_rows = db.scalars(
        select(CitizenReport)
        .where(CitizenReport.ward_id == ward_id)
        .order_by(CitizenReport.created_at.desc())
        .limit(120)
    ).all()
    civic_pressure = (
        float(sum(row.priority_weight for row in report_rows) / len(report_rows))
        if report_rows
        else 0.55
    )
    pressure_bonus = civic_pressure * 0.035
    category_boosts = _civic_priority_adjustments(report_rows)

    for row, score in zip(interventions, scores, strict=True):
        civic_category_bonus = category_boosts.get(row.category, 0.0)
        adjusted = float(score + pressure_bonus + civic_category_bonus + row.urgency * 0.01)
        row.ranking_score = round(adjusted, 6)
        row.justification = (
            f"{row.expected_beneficiaries} beneficiaries, "
            f"CI [{row.beneficiary_ci_low}, {row.beneficiary_ci_high}], "
            f"{row.estimated_cost_lakh:.2f} lakh cost, "
            f"impact/lakh {row.impact_per_lakh:.1f}, "
            f"equity {row.equity_need:.2f}, urgency {row.urgency:.2f}"
        )
    db.commit()

    ranked = sorted(interventions, key=lambda x: x.ranking_score, reverse=True)[:top_n]
    result = TopWorklistResponse(
        ward_id=ward_id,
        generated_at=datetime.now(timezone.utc),
        items=[_to_work_item(item) for item in ranked],
    )
    if _emit_audit:
        log_audit_event(
            db,
            action="top_worklist_generated",
            user=_,
            ward_id=ward_id,
            details={
                "top_n": top_n,
                "civic_pressure": round(civic_pressure, 4),
                "boosted_categories": list(category_boosts.keys()),
            },
        )
    return result


@router.get("/wards/{ward_id}/scenario", response_model=ScenarioResult)
def run_scenario(
    ward_id: int,
    budget_lakh: float = Query(default=8.0, gt=0.4, le=100.0),
    strategy: PlanningStrategy = Query(default="balanced"),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
    _emit_audit: bool = True,
) -> ScenarioResult:
    top_list = top_worklist(ward_id=ward_id, top_n=30, db=db, _=_, _emit_audit=False)
    item_dicts = [item.model_dump() for item in top_list.items]

    if not item_dicts:
        raise HTTPException(status_code=404, detail="No candidate projects found for simulation.")
    sim = simulate_policy_scenario(item_dicts, budget_lakh, strategy=strategy)
    selected_dicts = sim["selected_projects"]
    counterfactuals = build_counterfactuals(item_dicts, budget_lakh, strategy=strategy)
    result = ScenarioResult(
        ward_id=ward_id,
        budget_lakh=budget_lakh,
        strategy_profile=str(sim["strategy_profile"]),
        strategy_label=str(sim["strategy_label"]),
        strategy_description=str(sim["strategy_description"]),
        selected_projects=[TopWorkItem(**item) for item in selected_dicts],
        used_budget_lakh=float(sim["used_budget_lakh"]),
        remaining_budget_lakh=float(sim["remaining_budget_lakh"]),
        impacted_households=int(sim["impacted_households"]),
        estimated_sdg11_gain=float(sim["estimated_sdg11_gain"]),
        selection_method=str(sim["selection_method"]),
        decision_basis={
            **sim["decision_basis"],
            "budget_lakh": float(budget_lakh),
            "strategy_profile": str(strategy),
        },
        portfolio_summary=sim["portfolio_summary"],
        selected_reasoning=sim["selected_reasoning"],
        agency_load=sim["agency_load"],
        implementation_roadmap=sim["implementation_roadmap"],
        tradeoff_alerts=sim["tradeoff_alerts"],
        deferred_projects=sim["deferred_projects"],
        counterfactuals=counterfactuals,
        strategy_comparison=sim["strategy_comparison"],
    )
    if _emit_audit:
        log_audit_event(
            db,
            action="scenario_simulated",
            user=_,
            ward_id=ward_id,
            details={
                "budget_lakh": budget_lakh,
                "strategy_profile": strategy,
                "selected_projects": len(result.selected_projects),
                "impacted_households": result.impacted_households,
            },
        )
    return result


@router.get("/wards/{ward_id}/ai-budget-plan")
def ai_budget_plan(
    ward_id: int,
    budget_lakh: float = Query(default=50.0, gt=0.5, le=5000.0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    """AI-predicted sector-wise budget allocation based on live ward indicators."""
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    if not ward:
        raise HTTPException(status_code=404, detail="Ward not found.")
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    if not indicator:
        raise HTTPException(status_code=404, detail="Ward indicators not found.")

    map_layers = _load_json_file("ward_map_layers.json")
    layer_payload = map_layers.get(str(ward_id), {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}
    summary = layer_payload.get("summary", {})
    house_count = int(summary.get("houses", 0)) if isinstance(summary, dict) else 0
    road_count  = int(summary.get("roads", 0)) if isinstance(summary, dict) else 0

    indicators_dict = {
        "blocked_drain_count":  float(indicator.blocked_drain_count),
        "flood_exposure_pct":   float(indicator.flood_risk_index) * 100.0,
        "exposed_population":   float(indicator.exposed_population),
        "total_population":     float(ward.population) if ward.population else float(indicator.exposed_population),
        "informal_area_pct":    float(indicator.informal_area_pct),
        "green_deficit_index":  float(indicator.green_deficit_index),
        "road_length_km":       max(float(road_count) * 0.08, 0.1),
        "house_count":          float(house_count),
        "equity_score":         float(indicator.sdg11_score) / 10.0 if indicator.sdg11_score else 0.5,
    }

    result = predict_budget_allocation(indicators_dict, budget_lakh)
    log_audit_event(
        db,
        action="ai_budget_plan_generated",
        user=_,
        ward_id=ward_id,
        details={"budget_lakh": budget_lakh, "top_sector": result["top_sector"]},
    )
    return result


@router.post("/wards/{ward_id}/consult", response_model=ConsultResponse)
async def consult_ward_scenario(
    ward_id: int,
    req: ConsultRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConsultResponse:
    """AI Counselor: Translates vision into weights and runs simulation."""
    # Pre-defined deterministic responses for the frontend dropdown choices
    # This ensures 100% reliable and instantaneous results without relying on the LLM.
    preset_mappings = {
        "Tackle flood risks and improve drainage": {
            "weights": {"impact_per_lakh": 0.1, "equity_need": 0.2, "urgency": 0.35, "feasibility": 0.1, "beneficiary_norm": 0.15, "prior_rank_norm": 0.05, "readiness_norm": 0.05},
            "sector_priorities": {"Drainage": 1.0, "Water": 0.5, "Waste": 0.6, "Road": 0.4, "Green": 0.3, "Public Safety": 0.4},
            "sector_rationales": {"Drainage": "Directly prioritised by user to tackle flood risks and protect vulnerable households."},
            "reasoning": "Weighted heavily towards urgency and drainage to address flood risks.",
        },
        "Expand green spaces and reduce urban heat": {
            "weights": {"impact_per_lakh": 0.15, "equity_need": 0.15, "urgency": 0.1, "feasibility": 0.2, "beneficiary_norm": 0.2, "prior_rank_norm": 0.1, "readiness_norm": 0.1},
            "sector_priorities": {"Drainage": 0.4, "Water": 0.5, "Waste": 0.4, "Road": 0.3, "Green": 1.0, "Public Safety": 0.4},
            "sector_rationales": {"Green": "Directly prioritised by user to expand tree cover and mitigate urban heat effects."},
            "reasoning": "Weighted to focus on feasibility and beneficiary reach for green infrastructure.",
        },
        "Improve solid waste management": {
            "weights": {"impact_per_lakh": 0.2, "equity_need": 0.2, "urgency": 0.2, "feasibility": 0.15, "beneficiary_norm": 0.15, "prior_rank_norm": 0.05, "readiness_norm": 0.05},
            "sector_priorities": {"Drainage": 0.6, "Water": 0.4, "Waste": 1.0, "Road": 0.4, "Green": 0.4, "Public Safety": 0.4},
            "sector_rationales": {"Waste": "Directly prioritised by user to reduce disease vectors and clear waste backlogs."},
            "reasoning": "Balanced weights with a strong sector push towards solid waste management.",
        },
        "Ensure clean water access": {
            "weights": {"impact_per_lakh": 0.2, "equity_need": 0.3, "urgency": 0.2, "feasibility": 0.1, "beneficiary_norm": 0.1, "prior_rank_norm": 0.05, "readiness_norm": 0.05},
            "sector_priorities": {"Drainage": 0.4, "Water": 1.0, "Waste": 0.4, "Road": 0.3, "Green": 0.3, "Public Safety": 0.4},
            "sector_rationales": {"Water": "Directly prioritised by user to secure safe and continuous water access."},
            "reasoning": "Weighted heavily towards equity to ensure informal areas receive water infrastructure.",
        },
        "Repair and expand road networks": {
            "weights": {"impact_per_lakh": 0.25, "equity_need": 0.1, "urgency": 0.1, "feasibility": 0.2, "beneficiary_norm": 0.15, "prior_rank_norm": 0.1, "readiness_norm": 0.1},
            "sector_priorities": {"Drainage": 0.5, "Water": 0.4, "Waste": 0.4, "Road": 1.0, "Green": 0.3, "Public Safety": 0.5},
            "sector_rationales": {"Road": "Directly prioritised by user to improve connectivity and infrastructure gap."},
            "reasoning": "Weighted for maximum economic impact and feasibility of civil works.",
        },
        "Enhance public safety and lighting": {
            "weights": {"impact_per_lakh": 0.15, "equity_need": 0.25, "urgency": 0.2, "feasibility": 0.15, "beneficiary_norm": 0.1, "prior_rank_norm": 0.05, "readiness_norm": 0.1},
            "sector_priorities": {"Drainage": 0.4, "Water": 0.4, "Waste": 0.4, "Road": 0.5, "Green": 0.4, "Public Safety": 1.0},
            "sector_rationales": {"Public Safety": "Directly prioritised by user to improve community safety and street lighting."},
            "reasoning": "Weighted for equity and urgency to address immediate community safety needs.",
        },
        "Provide a balanced, data-driven approach": {
             "weights": {"impact_per_lakh": 0.2, "equity_need": 0.2, "urgency": 0.2, "feasibility": 0.1, "beneficiary_norm": 0.1, "prior_rank_norm": 0.1, "readiness_norm": 0.1},
             "sector_priorities": {},
             "sector_rationales": {},
             "reasoning": "Baseline data-driven approach. Budget is allocated strictly according to empirical ward indicators.",
        }
    }

    if req.sector_priorities is not None:
        # Questionnaire mode
        sector_priorities = req.sector_priorities
        custom_weights = {"impact_per_lakh": 0.2, "equity_need": 0.2, "urgency": 0.2, "feasibility": 0.1, "beneficiary_norm": 0.1, "prior_rank_norm": 0.1, "readiness_norm": 0.1}
        
        sector_rationales = {}
        if sector_priorities.get("Drainage", 0) > 0.1:
            sector_rationales["Drainage"] = "High drain blockage and flood exposure demand urgent drainage investment."
        if sector_priorities.get("Green", 0) > 0.1:
            sector_rationales["Green"] = "Green deficit index signals a shortfall in tree cover and open public space."
        if sector_priorities.get("Waste", 0) > 0.1:
            sector_rationales["Waste"] = "Dense informal housing correlates with solid waste management deficits."
        if sector_priorities.get("Water", 0) > 0.1:
            sector_rationales["Water"] = "Informal settlements and exposed population indicate acute water access gaps."
        if sector_priorities.get("Road", 0) > 0.1:
            sector_rationales["Road"] = "Infrastructure gap between household density and road coverage requires repair."
        if sector_priorities.get("Public Safety", 0) > 0.1:
            sector_rationales["Public Safety"] = "Baseline equity and community safety needs justify a minimum safety allocation."
            
        reasoning = "Custom questionnaire-driven priorities."
    elif req.vision in preset_mappings:
        mapping = preset_mappings[req.vision]
        custom_weights = mapping.get("weights", {})
        sector_priorities = mapping.get("sector_priorities", {})
        sector_rationales = mapping.get("sector_rationales", {})
        reasoning = mapping.get("reasoning", "")
    else:
        # Fallback to LLM if the frontend text was altered or free-typed
        consultation = await RAG_SERVICE.consult_scenario(req.vision)
        custom_weights = consultation.get("weights", {})
        sector_priorities = consultation.get("sector_priorities", {})
        sector_rationales = consultation.get("sector_rationales", {})
        reasoning = consultation.get("reasoning", "AI-optimized weights based on user vision.")

    # 2. Get candidate projects
    top_list = top_worklist(ward_id=ward_id, top_n=30, db=db, _=current_user, _emit_audit=False)
    item_dicts = [item.model_dump() for item in top_list.items]

    if not item_dicts:
        raise HTTPException(status_code=404, detail="No candidate projects found for simulation.")

    # 3. Handle budget plan results (sector distribution)
    # Re-use logic from ai_budget_plan to prepare indicators
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    
    # Defaults in case DB records are missing (fallback)
    house_count = 0
    road_count = 0
    if ward_id:
        try:
            map_layers = _load_json_file("ward_map_layers.json")
            layer_payload = map_layers.get(str(ward_id), {})
            if isinstance(layer_payload, dict):
                summary = layer_payload.get("summary", {})
                if isinstance(summary, dict):
                    house_count = int(summary.get("houses", 0))
                    road_count = int(summary.get("roads", 0))
        except:
            pass

    if indicator:
        indicators_dict = {
            "blocked_drain_count":  float(indicator.blocked_drain_count),
            "flood_exposure_pct":   float(indicator.flood_risk_index) * 100.0,
            "exposed_population":   float(indicator.exposed_population),
            "total_population":     float(ward.population) if (ward and ward.population) else float(indicator.exposed_population),
            "informal_area_pct":    float(indicator.informal_area_pct),
            "green_deficit_index":  float(indicator.green_deficit_index),
            "road_length_km":       max(float(road_count) * 0.08, 0.1),
            "house_count":          float(house_count),
            "equity_score":         float(indicator.sdg11_score) / 10.0 if indicator.sdg11_score else 0.5,
        }
    else:
        # Emergency fallback
        indicators_dict = {
            "blocked_drain_count": 0.0, "flood_exposure_pct": 20.0, "exposed_population": 1000.0,
            "total_population": 5000.0, "informal_area_pct": 10.0, "green_deficit_index": 0.5,
            "road_length_km": 1.0, "house_count": 500.0, "equity_score": 0.5
        }

    budget_plan = predict_budget_allocation(
        indicators_dict, 
        req.budget_lakh, 
        sector_priorities=sector_priorities,
        sector_rationales=sector_rationales
    )

    # 4. Run simulation with custom weights
    sim = simulate_policy_scenario(
        item_dicts, req.budget_lakh, strategy="custom", custom_weights=custom_weights
    )
    selected_dicts = sim["selected_projects"]
    counterfactuals = build_counterfactuals(
        item_dicts, req.budget_lakh, strategy="balanced", custom_weights=custom_weights
    )

    result = ScenarioResult(
        ward_id=ward_id,
        budget_lakh=req.budget_lakh,
        strategy_profile=str(sim["strategy_profile"]),
        strategy_label=str(sim["strategy_label"]),
        strategy_description=str(sim["strategy_description"]),
        selected_projects=[TopWorkItem(**item) for item in selected_dicts],
        used_budget_lakh=float(sim["used_budget_lakh"]),
        remaining_budget_lakh=float(sim["remaining_budget_lakh"]),
        impacted_households=int(sim["impacted_households"]),
        estimated_sdg11_gain=float(sim["estimated_sdg11_gain"]),
        selection_method=str(sim["selection_method"]),
        decision_basis={
            **sim["decision_basis"],
            "budget_lakh": float(req.budget_lakh),
            "vision": req.vision,
        },
        portfolio_summary=sim["portfolio_summary"],
        selected_reasoning=sim["selected_reasoning"],
        agency_load=sim["agency_load"],
        implementation_roadmap=sim["implementation_roadmap"],
        tradeoff_alerts=sim["tradeoff_alerts"],
        deferred_projects=sim["deferred_projects"],
        counterfactuals=counterfactuals,
        strategy_comparison=sim["strategy_comparison"],
    )

    log_audit_event(
        db,
        action="scenario_consulted",
        user=current_user,
        ward_id=ward_id,
        details={
            "vision": req.vision[:100],
            "budget_lakh": req.budget_lakh,
            "weights": custom_weights,
        },
    )

    return ConsultResponse(
        vision=req.vision,
        weights=custom_weights,
        reasoning=reasoning,
        result=result,
        budget_plan=budget_plan,
    )
