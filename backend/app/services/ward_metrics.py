from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_CITY_BBOX = (23.70, 90.32, 23.90, 90.52)  # lat_min, lon_min, lat_max, lon_max


@dataclass(frozen=True)
class WardMorphology:
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    population: int
    households: int
    road_length_km: float
    drain_length_km: float
    river_length_km: float
    water_area_km2: float
    house_footprint_km2: float
    park_area_km2: float
    playground_area_km2: float
    road_assets: int
    drain_assets: int
    river_assets: int
    water_assets: int
    green_nodes: int
    residential_features: int
    house_features: int
    green_features: int
    playground_features: int
    water_features: int
    waterway_features: int

    @property
    def green_area_km2(self) -> float:
        return float(self.park_area_km2 + 0.65 * self.playground_area_km2)


@dataclass(frozen=True)
class WardIndicatorScores:
    informal_area_pct: float
    blocked_drain_count: int
    green_deficit_index: float
    flood_risk_index: float
    sdg11_score: float
    exposed_population: int


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _path_length_km(path: list[list[float]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for p1, p2 in zip(path, path[1:]):
        if len(p1) != 2 or len(p2) != 2:
            continue
        lat1, lon1 = float(p1[0]), float(p1[1])
        lat2, lon2 = float(p2[0]), float(p2[1])
        mean_lat = (lat1 + lat2) / 2.0
        dx = (lon2 - lon1) * 111.0 * np.cos(np.radians(mean_lat))
        dy = (lat2 - lat1) * 111.0
        total += float((dx * dx + dy * dy) ** 0.5)
    return float(total)


def _polygon_area_km2(points: list[list[float]]) -> float:
    if len(points) < 4:
        return 0.0
    lats = np.array([float(p[0]) for p in points], dtype=np.float64)
    lons = np.array([float(p[1]) for p in points], dtype=np.float64)
    lat0 = np.mean(lats)
    xs = lons * 111.0 * np.cos(np.radians(lat0))
    ys = lats * 111.0
    area = 0.5 * np.abs(np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1]))
    return float(area)


def _boundary_centroid(boundary: list[list[float]]) -> tuple[float, float]:
    if not boundary:
        return 23.8103, 90.4125
    lats = [float(pt[0]) for pt in boundary if isinstance(pt, list) and len(pt) == 2]
    lons = [float(pt[1]) for pt in boundary if isinstance(pt, list) and len(pt) == 2]
    if not lats or not lons:
        return 23.8103, 90.4125
    return float(np.mean(lats)), float(np.mean(lons))


def estimate_population_and_households(
    area_km2: float,
    feature_counts: dict[str, int] | dict[str, float],
) -> tuple[int, int]:
    area = max(float(area_km2), 0.02)
    road_density = float(feature_counts.get("highway", 0.0)) / area
    residential_density = float(feature_counts.get("residential", 0.0)) / area
    house_density = float(feature_counts.get("house", 0.0)) / area
    population = int(
        np.clip(
            area * 12_500
            + residential_density * 30
            + road_density * 12
            + house_density * 36,
            12_000,
            260_000,
        )
    )
    households = max(int(population / 4.6), 1000)
    return population, households


def summarize_ward_morphology(
    boundary: list[list[float]],
    layers: dict[str, Any],
    feature_counts: dict[str, int] | dict[str, float],
    *,
    city_bbox: tuple[float, float, float, float] = DEFAULT_CITY_BBOX,
) -> WardMorphology:
    area_km2 = max(_polygon_area_km2(boundary), 0.02)
    centroid_lat, centroid_lon = _boundary_centroid(boundary)
    population, households = estimate_population_and_households(area_km2, feature_counts)

    def _paths(name: str) -> list[list[list[float]]]:
        payload = layers.get(name, [])
        if not isinstance(payload, list):
            return []
        return [path for path in payload if isinstance(path, list) and len(path) >= 2]

    def _points(name: str) -> list[dict[str, Any]]:
        payload = layers.get(name, [])
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    roads = _paths("roads")
    drains = _paths("drains")
    rivers = _paths("rivers")
    waters = _paths("waterbodies")
    houses = _points("houses")
    parks = _points("parks")
    playgrounds = _points("playgrounds")

    road_length_km = float(sum(_path_length_km(path) for path in roads))
    drain_length_km = float(sum(_path_length_km(path) for path in drains))
    river_length_km = float(sum(_path_length_km(path) for path in rivers))
    water_area_km2 = float(sum(_polygon_area_km2(path) for path in waters))
    house_footprint_km2 = float(sum(float(item.get("footprint", 0.0)) for item in houses))
    park_area_km2 = float(
        sum(
            float(item.get("area_km2", 0.0))
            if isinstance(item.get("area_km2"), (int, float))
            else max(float(item.get("size", 0.0)) / 80.0, 0.0)
            for item in parks
        )
    )
    playground_area_km2 = float(
        sum(
            float(item.get("area_km2", 0.0))
            if isinstance(item.get("area_km2"), (int, float))
            else max(float(item.get("size", 0.0)) / 80.0, 0.0)
            for item in playgrounds
        )
    )

    lat_min, lon_min, lat_max, lon_max = city_bbox
    centroid_lat = float(np.clip(centroid_lat, lat_min, lat_max))
    centroid_lon = float(np.clip(centroid_lon, lon_min, lon_max))

    return WardMorphology(
        area_km2=area_km2,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        population=population,
        households=households,
        road_length_km=road_length_km,
        drain_length_km=drain_length_km,
        river_length_km=river_length_km,
        water_area_km2=water_area_km2,
        house_footprint_km2=house_footprint_km2,
        park_area_km2=park_area_km2,
        playground_area_km2=playground_area_km2,
        road_assets=len(roads),
        drain_assets=len(drains),
        river_assets=len(rivers),
        water_assets=len(waters),
        green_nodes=len(parks) + len(playgrounds),
        residential_features=int(float(feature_counts.get("residential", 0.0))),
        house_features=int(float(feature_counts.get("house", 0.0))),
        green_features=int(float(feature_counts.get("green", 0.0))),
        playground_features=int(float(feature_counts.get("playground", 0.0))),
        water_features=int(float(feature_counts.get("water", 0.0))),
        waterway_features=int(float(feature_counts.get("waterway", 0.0))),
    )


def estimate_blocked_network_scale(morphology: WardMorphology) -> float:
    extent = (
        3.5
        + morphology.road_length_km * 5.0
        + (morphology.drain_length_km + morphology.river_length_km * 0.7) * 7.2
        + morphology.road_assets * 0.08
    )
    if morphology.drain_assets + morphology.river_assets == 0:
        extent *= 0.72
    return float(np.clip(extent, 2.0, 180.0))


def estimate_exposed_population(
    population: int,
    flood_risk_index: float,
    blocked_drain_count: int,
    informal_area_pct: float,
    morphology: WardMorphology,
) -> int:
    blocked_ratio = float(
        np.clip(blocked_drain_count / max(estimate_blocked_network_scale(morphology), 1.0), 0.0, 1.0)
    )
    exposure_ratio = np.clip(
        0.05
        + float(flood_risk_index) * 0.42
        + blocked_ratio * 0.18
        + float(informal_area_pct) / 100.0 * 0.17,
        0.05,
        0.92,
    )
    return int(np.clip(round(int(population) * exposure_ratio), 500, int(population)))


def score_sdg11(
    *,
    flood_risk_index: float,
    green_deficit_index: float,
    blocked_drain_count: int,
    informal_area_pct: float,
    morphology: WardMorphology,
) -> float:
    blocked_ratio = float(
        np.clip(blocked_drain_count / max(estimate_blocked_network_scale(morphology), 1.0), 0.0, 1.0)
    )
    stress = (
        0.33 * float(flood_risk_index)
        + 0.25 * float(green_deficit_index)
        + 0.18 * blocked_ratio
        + 0.24 * (float(informal_area_pct) / 100.0)
    )
    return float(np.clip(92.0 - stress * 56.0 - min(int(blocked_drain_count), 180) * 0.05, 18.0, 91.0))


def score_live_flood_risk(
    *,
    baseline_flood_risk: float,
    blocked_drain_count: int,
    green_deficit_index: float,
    informal_area_pct: float,
    morphology: WardMorphology,
    rainfall_mm: float | None,
    blocked_signals: int,
    sensor_pulses: int,
) -> float:
    blocked_ratio = float(
        np.clip(blocked_drain_count / max(estimate_blocked_network_scale(morphology), 1.0), 0.0, 1.0)
    )
    rain_pressure = float(np.clip((rainfall_mm or 0.0) / 95.0, 0.0, 1.0))
    complaint_pressure = float(np.clip(blocked_signals / 18.0, 0.0, 1.0))
    sensor_pressure = float(np.clip(sensor_pulses / 16.0, 0.0, 1.0))
    risk = (
        0.56 * float(baseline_flood_risk)
        + 0.18 * blocked_ratio
        + 0.10 * float(green_deficit_index)
        + 0.07 * (float(informal_area_pct) / 100.0)
        + 0.05 * rain_pressure
        + 0.03 * complaint_pressure
        + 0.01 * sensor_pressure
    )
    return float(np.clip(risk, 0.03, 0.98))


def _robust_scale(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if float(np.nanmax(arr) - np.nanmin(arr)) < 1e-9:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    lo = float(np.quantile(arr, 0.12))
    hi = float(np.quantile(arr, 0.88))
    if hi - lo < 1e-9:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def score_ward_indicators(morphologies: dict[int, WardMorphology]) -> dict[int, WardIndicatorScores]:
    if not morphologies:
        return {}

    ordered_ids = list(morphologies.keys())
    rows = [morphologies[ward_id] for ward_id in ordered_ids]

    area = np.array([row.area_km2 for row in rows], dtype=np.float64)
    green_share = np.array([row.green_area_km2 / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    green_nodes_per_1k = np.array([row.green_nodes / max(row.households, 1) * 1000.0 for row in rows], dtype=np.float64)
    green_m2_per_resident = np.array(
        [row.green_area_km2 * 1_000_000.0 / max(row.population, 1) for row in rows],
        dtype=np.float64,
    )
    impervious_ratio = np.array([row.house_footprint_km2 / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    road_density = np.array([row.road_length_km / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    house_density = np.array([row.house_features / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    population_density = np.array([row.population / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    water_surface_ratio = np.array([row.water_area_km2 / max(row.area_km2, 0.02) for row in rows], dtype=np.float64)
    water_feature_density = np.array(
        [(row.water_features + 0.7 * row.waterway_features) / max(row.area_km2, 0.02) for row in rows],
        dtype=np.float64,
    )
    drain_capacity_raw = np.array(
        [
            (row.drain_length_km + row.river_length_km * 0.55 + row.drain_assets * 0.025)
            / (row.road_length_km + row.house_features * 0.0025 + 0.35)
            for row in rows
        ],
        dtype=np.float64,
    )
    southness = np.array(
        [
            1.0 - ((row.centroid_lat - DEFAULT_CITY_BBOX[0]) / max(DEFAULT_CITY_BBOX[2] - DEFAULT_CITY_BBOX[0], 1e-6))
            for row in rows
        ],
        dtype=np.float64,
    )
    eastness = np.array(
        [
            (row.centroid_lon - DEFAULT_CITY_BBOX[1]) / max(DEFAULT_CITY_BBOX[3] - DEFAULT_CITY_BBOX[1], 1e-6)
            for row in rows
        ],
        dtype=np.float64,
    )
    lowland_raw = 0.62 * np.clip(southness, 0.0, 1.0) + 0.38 * np.clip(eastness, 0.0, 1.0)

    green_supply = (
        0.45 * _robust_scale(np.log1p(green_m2_per_resident))
        + 0.30 * _robust_scale(np.log1p(green_nodes_per_1k))
        + 0.25 * _robust_scale(green_share)
    )
    impervious_pressure = (
        0.42 * _robust_scale(impervious_ratio)
        + 0.33 * _robust_scale(road_density)
        + 0.25 * _robust_scale(house_density)
    )
    population_pressure = _robust_scale(population_density)
    water_hazard = 0.58 * _robust_scale(water_surface_ratio) + 0.42 * _robust_scale(water_feature_density)
    drain_capacity = _robust_scale(drain_capacity_raw)
    lowland = _robust_scale(lowland_raw)

    green_deficit = np.clip(
        0.08
        + 0.84
        * (
            0.66 * (1.0 - green_supply)
            + 0.22 * impervious_pressure
            + 0.12 * population_pressure
        ),
        0.05,
        0.96,
    )

    drain_pressure = (
        0.38 * impervious_pressure
        + 0.24 * population_pressure
        + 0.24 * water_hazard
        + 0.14 * lowland
    )
    blocked_ratio = np.clip(
        0.04 + 0.90 * (0.58 * drain_pressure + 0.42 * (1.0 - drain_capacity)),
        0.03,
        0.97,
    )

    flood_risk = np.clip(
        0.04
        + 0.56
        * (
            0.30 * water_hazard
            + 0.24 * blocked_ratio
            + 0.18 * impervious_pressure
            + 0.14 * green_deficit
            + 0.08 * lowland
            + 0.06 * (1.0 - drain_capacity)
        ),
        0.03,
        0.96,
    )

    informal_area = np.clip(
        4.5
        + flood_risk * 31.0
        + green_deficit * 17.0
        + impervious_pressure * 16.0
        + population_pressure * 8.0
        + blocked_ratio * 7.5,
        4.0,
        82.0,
    )

    scores: dict[int, WardIndicatorScores] = {}
    for idx, ward_id in enumerate(ordered_ids):
        morphology = rows[idx]
        network_scale = estimate_blocked_network_scale(morphology)
        blocked_count = int(round(float(np.clip(blocked_ratio[idx] * network_scale, 0.0, 180.0))))
        if morphology.road_assets + morphology.drain_assets + morphology.river_assets == 0:
            blocked_count = 0

        informal_pct = float(round(float(informal_area[idx]), 2))
        flood_index = float(round(float(flood_risk[idx]), 4))
        green_index = float(round(float(green_deficit[idx]), 4))
        exposed_population = estimate_exposed_population(
            morphology.population,
            flood_index,
            blocked_count,
            informal_pct,
            morphology,
        )
        sdg11_score = score_sdg11(
            flood_risk_index=flood_index,
            green_deficit_index=green_index,
            blocked_drain_count=blocked_count,
            informal_area_pct=informal_pct,
            morphology=morphology,
        )
        scores[ward_id] = WardIndicatorScores(
            informal_area_pct=informal_pct,
            blocked_drain_count=blocked_count,
            green_deficit_index=green_index,
            flood_risk_index=flood_index,
            sdg11_score=float(round(sdg11_score, 2)),
            exposed_population=exposed_population,
        )
    return scores
