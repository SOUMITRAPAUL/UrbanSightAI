from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.spatial import Voronoi

from app.core.config import PROCESSED_DIR, RAW_DIR, REAL_DATA_ONLY
from app.services.ward_metrics import (
    estimate_blocked_network_scale,
    score_ward_indicators,
    summarize_ward_morphology,
)


DATASET_VERSION = "2.8.0"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_ALT_URL = "https://overpass.kumi.systems/api/interpreter"
DHAKA_BBOX = (23.70, 90.32, 23.90, 90.52)  # lat_min, lon_min, lat_max, lon_max
DHAKA_EXT_BBOX = (23.65, 90.25, 23.95, 90.55)
DHAKA_METRO_REL_ID = 13663697
DHAKA_CENTER = (23.8103, 90.4125)

HDX_ADMIN_GEOJSON_ZIP_URL = (
    "https://data.humdata.org/dataset/401d3fae-4262-48c9-891f-461fd776d49b/"
    "resource/cec2abe3-d8b7-4025-9362-9f7e780f2a07/download/bgd_admin_boundaries.geojson.zip"
)
HDX_ADMIN_POINTS_GEOJSON = "bgd_adminpoints.geojson"
HDX_DHAKA_WARD_RAW_CACHE = "dhaka_hdx_ward_boundaries.json"
WARD_NAME_PATTERN = re.compile(r"ward\s*no[- ]?(\d+)", re.IGNORECASE)
DHAKA_CITY_THANAS = {
    "Adabor",
    "Badda",
    "Bangshal",
    "Biman Bandar",
    "Cantonment",
    "Chak Bazar",
    "Dakshinkhan",
    "Darus Salam",
    "Demra",
    "Dhanmondi",
    "Gendaria",
    "Gulshan",
    "Hazaribagh",
    "Jatrabari",
    "Kadamtali",
    "Kafrul",
    "Kalabagan",
    "Kamrangir Char",
    "Khilgaon",
    "Khilkhet",
    "Kotwali",
    "Lalbagh",
    "Mirpur",
    "Mohammadpur",
    "Motijheel",
    "New Market",
    "Pallabi",
    "Paltan",
    "Ramna",
    "Rampura",
    "Sabujbagh",
    "Shah Ali",
    "Shahbagh",
    "Sher-e-bangla Nagar",
    "Shyampur",
    "Sutrapur",
    "Tejgaon",
    "Tejgaon Ind. Area",
    "Turag",
    "Uttar Khan",
    "Uttara",
}


def _ensure_closed(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return points
    if points[0] != points[-1]:
        return [*points, points[0]]
    return points


def _polygon_area_km2(points: list[list[float]]) -> float:
    if len(points) < 4:
        return 0.0
    lats = np.array([p[0] for p in points], dtype=np.float64)
    lons = np.array([p[1] for p in points], dtype=np.float64)
    lat0 = np.mean(lats)
    xs = lons * 111.0 * np.cos(np.radians(lat0))
    ys = lats * 111.0
    area = 0.5 * np.abs(np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1]))
    return float(area)


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return min(lats), min(lons), max(lats), max(lons)


def _point_in_polygon(lat: float, lon: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 4:
        return False
    inside = False
    x, y = lon, lat
    for i in range(len(polygon) - 1):
        y1, x1 = polygon[i]
        y2, x2 = polygon[i + 1]
        intersects = (y1 > y) != (y2 > y)
        if not intersects:
            continue
        slope_x = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
        if x < slope_x:
            inside = not inside
    return inside


def _convex_hull(points: list[list[float]]) -> list[list[float]]:
    unique = sorted({(p[1], p[0]) for p in points})
    if len(unique) < 3:
        return [[lat, lon] for lon, lat in unique]

    def cross(
        o: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    latlon = [[lat, lon] for lon, lat in hull]
    return _ensure_closed(latlon)


@dataclass
class WardPolygon:
    ward_id: int
    code: str
    name: str
    boundary: list[list[float]]
    _bbox: tuple[float, float, float, float] | None = None
    _area_km2: float | None = None

    def __post_init__(self) -> None:
        self.boundary = _ensure_closed(self.boundary)
        self._bbox = _bbox(self.boundary) if self.boundary else (0.0, 0.0, 0.0, 0.0)
        self._area_km2 = _polygon_area_km2(self.boundary)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox if self._bbox is not None else (0.0, 0.0, 0.0, 0.0)

    @property
    def area_km2(self) -> float:
        return float(self._area_km2 if self._area_km2 is not None else 0.0)

    def contains(self, lat: float, lon: float) -> bool:
        lat_min, lon_min, lat_max, lon_max = self.bbox
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            return False
        return _point_in_polygon(lat, lon, self.boundary)


@dataclass
class WardPoint:
    ward_number: int
    thana: str
    pcode: str
    lat: float
    lon: float
    original_name: str
    is_partial: bool


def _overpass_fetch(query: str, timeout: int = 180) -> dict[str, Any]:
    last_exc: Exception | None = None
    for endpoint in [OVERPASS_ALT_URL, OVERPASS_URL]:
        try:
            response = requests.post(endpoint, data={"data": query}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("elements"), list):
                return payload
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError("Overpass query failed across all configured endpoints.") from last_exc


def _extract_relation_polygon(element: dict[str, Any]) -> list[list[float]]:
    members = element.get("members", [])
    outers = [
        m.get("geometry", [])
        for m in members
        if m.get("type") == "way" and m.get("role", "") in {"", "outer"}
    ]
    outers = [geom for geom in outers if geom]
    if not outers:
        return []
    if len(outers) == 1:
        points = [[float(p["lat"]), float(p["lon"])] for p in outers[0]]
        return _ensure_closed(points)

    all_points = [
        [float(p["lat"]), float(p["lon"])] for geom in outers for p in geom if "lat" in p and "lon" in p
    ]
    return _convex_hull(all_points)


def _extract_way_polygon(element: dict[str, Any]) -> list[list[float]]:
    geometry = element.get("geometry", [])
    points = [[float(p["lat"]), float(p["lon"])] for p in geometry if "lat" in p and "lon" in p]
    return _ensure_closed(points)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _parse_ward_number(name: str) -> int | None:
    match = WARD_NAME_PATTERN.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _load_hdx_adminpoints(raw_dir: Path) -> dict[str, Any]:
    cache_path = raw_dir / HDX_ADMIN_POINTS_GEOJSON
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload.get("features"), list) and len(payload["features"]) >= 1000:
                return payload
        except Exception:
            pass

    last_exc: Exception | None = None
    for url in [
        HDX_ADMIN_GEOJSON_ZIP_URL,
        HDX_ADMIN_GEOJSON_ZIP_URL.replace("/download/", "/down/"),
    ]:
        try:
            response = requests.get(url, timeout=280)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                with archive.open(HDX_ADMIN_POINTS_GEOJSON) as src:
                    payload = json.load(io.TextIOWrapper(src, encoding="utf-8"))
            with cache_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=True, indent=2)
            return payload
        except Exception as exc:
            last_exc = exc

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    raise RuntimeError(
        "Unable to load HDX Bangladesh admin points dataset for ward extraction."
    ) from last_exc


def _extract_hdx_dhaka_ward_points(payload: dict[str, Any]) -> list[WardPoint]:
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []

    points: list[WardPoint] = []
    seen_pcodes: set[str] = set()
    for feature in features:
        properties = feature.get("properties", {})
        if int(properties.get("admin_level", -1)) != 4:
            continue

        adm2_name = _normalize_text(properties.get("adm2_name", ""))
        if adm2_name.casefold() != "dhaka":
            continue

        thana = _normalize_text(properties.get("adm3_name", ""))
        if thana not in DHAKA_CITY_THANAS:
            continue

        name_raw = _normalize_text(properties.get("adm4_name") or properties.get("name") or "")
        ward_number = _parse_ward_number(name_raw)
        if ward_number is None:
            continue

        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        lon = float(coordinates[0])
        lat = float(coordinates[1])
        pcode = str(properties.get("adm4_pcode", "")).strip()
        if not pcode or pcode in seen_pcodes:
            continue
        seen_pcodes.add(pcode)

        is_partial = "part" in name_raw.casefold() or "rest" in name_raw.casefold()
        points.append(
            WardPoint(
                ward_number=ward_number,
                thana=thana,
                pcode=pcode,
                lat=lat,
                lon=lon,
                original_name=name_raw,
                is_partial=is_partial,
            )
        )

    points.sort(
        key=lambda point: (
            point.ward_number,
            1 if point.is_partial else 0,
            abs(point.lat - DHAKA_CENTER[0]) + abs(point.lon - DHAKA_CENTER[1]),
            point.pcode,
        )
    )
    return points


def _voronoi_finite_polygons_2d(
    vor: Voronoi,
    radius: float | None = None,
) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 3.0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_1, point_2), (vertex_1, vertex_2) in zip(
        vor.ridge_points,
        vor.ridge_vertices,
        strict=True,
    ):
        all_ridges.setdefault(point_1, []).append((point_2, vertex_1, vertex_2))
        all_ridges.setdefault(point_2, []).append((point_1, vertex_1, vertex_2))

    for point_index, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(list(vertices))
            continue

        ridges = all_ridges.get(point_index, [])
        new_region = [vertex for vertex in vertices if vertex >= 0]

        for neighbor, vertex_1, vertex_2 in ridges:
            if vertex_2 < 0:
                vertex_1, vertex_2 = vertex_2, vertex_1
            if vertex_1 >= 0:
                continue

            tangent = vor.points[neighbor] - vor.points[point_index]
            tangent /= np.linalg.norm(tangent) + 1e-12
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, neighbor]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex_2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        region_vertices = np.asarray([new_vertices[v] for v in new_region])
        region_center = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - region_center[1],
            region_vertices[:, 0] - region_center[0],
        )
        new_region = np.asarray(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _clip_polygon_with_edge(
    polygon: list[tuple[float, float]],
    inside_fn: Any,
    intersect_fn: Any,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    prev = polygon[-1]
    prev_inside = inside_fn(prev)

    for curr in polygon:
        curr_inside = inside_fn(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect_fn(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect_fn(prev, curr))
        prev = curr
        prev_inside = curr_inside

    return output


def _intersect_vertical(
    p1: tuple[float, float],
    p2: tuple[float, float],
    x_edge: float,
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) < 1e-12:
        return x_edge, y1
    ratio = (x_edge - x1) / (x2 - x1)
    return x_edge, y1 + ratio * (y2 - y1)


def _intersect_horizontal(
    p1: tuple[float, float],
    p2: tuple[float, float],
    y_edge: float,
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1e-12:
        return x1, y_edge
    ratio = (y_edge - y1) / (y2 - y1)
    return x1 + ratio * (x2 - x1), y_edge


def _clip_polygon_to_bbox(
    polygon_xy: list[tuple[float, float]],
    bbox_latlon: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    lat_min, lon_min, lat_max, lon_max = bbox_latlon
    clipped = list(polygon_xy)
    clipped = _clip_polygon_with_edge(
        clipped,
        lambda p: p[0] >= lon_min,
        lambda p1, p2: _intersect_vertical(p1, p2, lon_min),
    )
    clipped = _clip_polygon_with_edge(
        clipped,
        lambda p: p[0] <= lon_max,
        lambda p1, p2: _intersect_vertical(p1, p2, lon_max),
    )
    clipped = _clip_polygon_with_edge(
        clipped,
        lambda p: p[1] >= lat_min,
        lambda p1, p2: _intersect_horizontal(p1, p2, lat_min),
    )
    clipped = _clip_polygon_with_edge(
        clipped,
        lambda p: p[1] <= lat_max,
        lambda p1, p2: _intersect_horizontal(p1, p2, lat_max),
    )

    unique: list[tuple[float, float]] = []
    for x, y in clipped:
        if not unique or abs(unique[-1][0] - x) > 1e-9 or abs(unique[-1][1] - y) > 1e-9:
            unique.append((float(x), float(y)))
    if len(unique) >= 2:
        first_x, first_y = unique[0]
        last_x, last_y = unique[-1]
        if abs(first_x - last_x) < 1e-9 and abs(first_y - last_y) < 1e-9:
            unique.pop()
    return unique


def _voronoi_guard_points(
    bbox_latlon: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    lat_min, lon_min, lat_max, lon_max = bbox_latlon
    lat_pad = max((lat_max - lat_min) * 0.08, 0.01)
    lon_pad = max((lon_max - lon_min) * 0.08, 0.01)
    lats = np.linspace(lat_min - lat_pad, lat_max + lat_pad, num=11)
    lons = np.linspace(lon_min - lon_pad, lon_max + lon_pad, num=11)

    points: list[tuple[float, float]] = []
    for lon in lons:
        points.append((float(lon), float(lat_min - lat_pad)))
        points.append((float(lon), float(lat_max + lat_pad)))
    for lat in lats[1:-1]:
        points.append((float(lon_min - lon_pad), float(lat)))
        points.append((float(lon_max + lon_pad), float(lat)))
    return points


def _load_cached_hdx_ward_boundaries(raw_dir: Path) -> list[WardPolygon] | None:
    path = raw_dir / HDX_DHAKA_WARD_RAW_CACHE
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records = payload.get("wards", []) if isinstance(payload, dict) else []
        wards: list[WardPolygon] = []
        for item in records:
            boundary = item.get("boundary", [])
            if not isinstance(boundary, list) or len(boundary) < 4:
                continue
            wards.append(
                WardPolygon(
                    ward_id=int(item.get("ward_id", len(wards) + 1)),
                    code=str(item.get("code", f"DHK-W{len(wards) + 1:02d}")),
                    name=str(item.get("name", f"Ward {len(wards) + 1}")),
                    boundary=[[float(pt[0]), float(pt[1])] for pt in boundary if len(pt) == 2],
                )
            )
        if len(wards) >= 20:
            return wards
    except Exception:
        return None
    return None


def _build_hdx_ward_boundaries(raw_dir: Path) -> list[WardPolygon]:
    cached = _load_cached_hdx_ward_boundaries(raw_dir)
    if cached is not None:
        return cached

    payload = _load_hdx_adminpoints(raw_dir)
    ward_points = _extract_hdx_dhaka_ward_points(payload)
    if len(ward_points) < 20:
        raise RuntimeError("HDX ward point extraction returned insufficient Dhaka ward points.")

    guard_points = _voronoi_guard_points(DHAKA_BBOX)
    points_xy = np.array(
        [[point.lon, point.lat] for point in ward_points] + guard_points,
        dtype=np.float64,
    )
    vor = Voronoi(points_xy)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    wards: list[WardPolygon] = []
    for idx, point in enumerate(ward_points, start=1):
        region_indices = regions[idx - 1]
        polygon_xy = [
            (float(vertices[v][0]), float(vertices[v][1]))
            for v in region_indices
        ]
        clipped_xy = _clip_polygon_to_bbox(polygon_xy, DHAKA_BBOX)
        if len(clipped_xy) < 3:
            continue
        boundary = _ensure_closed([[round(lat, 6), round(lon, 6)] for lon, lat in clipped_xy])
        if _polygon_area_km2(boundary) < 0.004:
            continue

        label = point.original_name if point.original_name else f"Ward No-{point.ward_number:02d}"
        name = f"{label} - {point.thana}"
        wards.append(
            WardPolygon(
                ward_id=len(wards) + 1,
                code=f"DHK-{point.pcode}",
                name=name,
                boundary=boundary,
            )
        )

    if len(wards) < 20:
        raise RuntimeError("Voronoi polygonization produced too few Dhaka ward boundaries.")

    with (raw_dir / HDX_DHAKA_WARD_RAW_CACHE).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source": "hdx_codab_admin4_points_voronoi",
        "generated_at": datetime.now(timezone.utc).isoformat(),
                "wards": [
                    {
                        "ward_id": ward.ward_id,
                        "code": ward.code,
                        "name": ward.name,
                        "boundary": ward.boundary,
                    }
                    for ward in wards
                ],
            },
            file,
            ensure_ascii=True,
            indent=2,
        )
    return wards


def _build_fallback_grid(rows: int = 3, cols: int = 3) -> list[WardPolygon]:
    lat_min, lon_min, lat_max, lon_max = DHAKA_BBOX
    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols

    wards: list[WardPolygon] = []
    ward_id = 1
    for row in range(rows):
        for col in range(cols):
            r0 = lat_min + row * lat_step
            r1 = r0 + lat_step
            c0 = lon_min + col * lon_step
            c1 = c0 + lon_step
            boundary = _ensure_closed(
                [
                    [r0, c0],
                    [r0, c1],
                    [r1, c1],
                    [r1, c0],
                ]
            )
            wards.append(
                WardPolygon(
                    ward_id=ward_id,
                    code=f"DHK-W{ward_id:02d}",
                    name=f"Synthetic Zone {ward_id}",
                    boundary=boundary,
                )
            )
            ward_id += 1
    return wards


def _assign_codes(wards: list[WardPolygon]) -> list[WardPolygon]:
    ordered = sorted(wards, key=lambda w: (-w.area_km2, w.name.lower()))
    for idx, ward in enumerate(ordered, start=1):
        ward.ward_id = idx
        ward.code = f"DHK-W{idx:02d}"
    return ordered


def _parse_boundary_elements(
    elements: list[dict[str, Any]],
    min_area_km2: float,
) -> list[WardPolygon]:
    wards: list[WardPolygon] = []
    seen_names: set[str] = set()

    for element in elements:
        tags = element.get("tags", {})
        name = str(tags.get("name", "")).strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen_names:
            continue

        if element.get("type") == "way":
            boundary = _extract_way_polygon(element)
        elif element.get("type") == "relation":
            boundary = _extract_relation_polygon(element)
        else:
            continue

        if len(boundary) < 4:
            continue
        area = _polygon_area_km2(boundary)
        if area < min_area_km2:
            continue

        seen_names.add(key)
        wards.append(
            WardPolygon(
                ward_id=0,
                code="",
                name=name,
                boundary=boundary,
            )
        )

    return _assign_codes(wards)


def _load_cached_boundaries(raw_dir: Path) -> tuple[list[WardPolygon], str] | None:
    hdx_cached = _load_cached_hdx_ward_boundaries(raw_dir)
    if hdx_cached is not None:
        return hdx_cached, "hdx_codab_admin4_points_voronoi_cache"

    candidates = [
        ("dhaka_admin_ward_boundaries.json", "osm_admin_level_10_cache", 0.08, 3),
    ]
    for filename, source, min_area, min_count in candidates:
        path = raw_dir / filename
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            wards = _parse_boundary_elements(payload.get("elements", []), min_area_km2=min_area)
            if len(wards) >= min_count:
                return wards, source
        except Exception:
            continue
    return None


def _download_ward_boundaries(raw_dir: Path) -> tuple[list[WardPolygon], str]:
    hdx_exc: Exception | None = None
    try:
        hdx_wards = _build_hdx_ward_boundaries(raw_dir)
        if len(hdx_wards) >= 20:
            return hdx_wards, "hdx_codab_admin4_points_voronoi"
    except Exception as exc:
        hdx_exc = exc

    cached = _load_cached_boundaries(raw_dir)
    if cached:
        return cached

    ext_lat_min, ext_lon_min, ext_lat_max, ext_lon_max = DHAKA_EXT_BBOX

    admin_query = f"""
[out:json][timeout:180];
(
  relation["boundary"="administrative"]["admin_level"="10"]["name"]({ext_lat_min},{ext_lon_min},{ext_lat_max},{ext_lon_max});
  relation({DHAKA_METRO_REL_ID});
  map_to_area->.metro;
  relation(area.metro)["boundary"="administrative"]["admin_level"="10"]["name"];
);
out geom;
"""

    try:
        admin_payload = _overpass_fetch(admin_query)
        with (raw_dir / "dhaka_admin_ward_boundaries.json").open("w", encoding="utf-8") as file:
            json.dump(admin_payload, file, ensure_ascii=True, indent=2)
        admin_wards = _parse_boundary_elements(admin_payload.get("elements", []), min_area_km2=0.08)
        if len(admin_wards) >= 3:
            return admin_wards, "osm_admin_level_10"
    except Exception as exc:
        cached = _load_cached_boundaries(raw_dir)
        if cached:
            return cached
        if REAL_DATA_ONLY:
            raise RuntimeError(
                "Real-data mode is enabled: unable to resolve Dhaka ward boundaries from HDX/OSM and no cache is available."
            ) from exc

    if REAL_DATA_ONLY:
        if hdx_exc is not None:
            raise RuntimeError(
                "Real-data mode is enabled: HDX Dhaka ward extraction failed and OSM admin ward fallback was unavailable."
            ) from hdx_exc
        raise RuntimeError(
            "Real-data mode is enabled: no trusted Dhaka ward boundaries available from API or cache."
        )

    return _build_fallback_grid(rows=3, cols=3), "synthetic_grid_fallback"


def _overpass_feature_query() -> str:
    lat_min, lon_min, lat_max, lon_max = DHAKA_BBOX
    return f"""
[out:json][timeout:180];
(
  way["waterway"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["landuse"="residential"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["landuse"="grass"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["leisure"="park"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["leisure"="playground"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""


def _overpass_building_query() -> str:
    lat_min, lon_min, lat_max, lon_max = DHAKA_BBOX
    return f"""
[out:json][timeout:180];
(
  way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center;
"""


def _feature_type(tags: dict[str, Any]) -> str | None:
    if tags.get("waterway"):
        return "waterway"
    if tags.get("highway"):
        return "highway"
    if tags.get("natural") == "water":
        return "water"
    if tags.get("building"):
        return "house"
    if tags.get("leisure") == "playground":
        return "playground"
    if tags.get("landuse") == "residential":
        return "residential"
    if tags.get("landuse") == "grass" or tags.get("leisure") == "park":
        return "green"
    return None


def _load_cached_osm_features(raw_dir: Path) -> list[dict[str, Any]] | None:
    path = raw_dir / "dhaka_osm_features.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        elements = payload.get("elements", [])
        if isinstance(elements, list) and len(elements) > 500:
            return elements
    except Exception:
        return None
    return None


def _download_osm_features(raw_dir: Path, rng: np.random.Generator) -> list[dict[str, Any]]:
    cached_elements = _load_cached_osm_features(raw_dir)
    if cached_elements is not None:
        return cached_elements

    query = _overpass_feature_query()
    building_query = _overpass_building_query()
    try:
        payload = _overpass_fetch(query)
        elements = payload.get("elements", [])

        building_payload = _overpass_fetch(building_query)
        with (raw_dir / "dhaka_building_centers.json").open("w", encoding="utf-8") as file:
            json.dump(building_payload, file, ensure_ascii=True, indent=2)
        building_elements = []
        for item in building_payload.get("elements", []):
            center = item.get("center")
            if not center:
                continue
            item_id = int(item.get("id", 0))
            # Keep a deterministic sample for pilot-scale rendering performance.
            if item_id % 6 != 0:
                continue
            building_elements.append(
                {
                    "id": item_id,
                    "tags": {"building": item.get("tags", {}).get("building", "yes")},
                    "geometry": [
                        {
                            "lat": float(center.get("lat")),
                            "lon": float(center.get("lon")),
                        }
                    ],
                }
            )

        elements.extend(building_elements)
        with (raw_dir / "dhaka_osm_features.json").open("w", encoding="utf-8") as file:
            json.dump({"elements": elements}, file, ensure_ascii=True, indent=2)
        return elements
    except Exception as exc:
        cached_elements = _load_cached_osm_features(raw_dir)
        if cached_elements is not None:
            return cached_elements

        if REAL_DATA_ONLY:
            raise RuntimeError(
                "Real-data mode is enabled: unable to fetch OSM map features and no cache is available."
            ) from exc

        fallback = []
        for i in range(380):
            tags: dict[str, str] = {}
            dice = rng.random()
            if dice < 0.22:
                tags["waterway"] = "drain"
            elif dice < 0.52:
                tags["highway"] = "residential"
            elif dice < 0.68:
                tags["building"] = "yes"
            elif dice < 0.82:
                tags["landuse"] = "residential"
            elif dice < 0.9:
                tags["leisure"] = "playground"
            else:
                tags["landuse"] = "grass"

            fallback.append(
                {
                    "id": i + 1,
                    "tags": tags,
                    "geometry": [
                        {
                            "lat": float(rng.uniform(DHAKA_BBOX[0], DHAKA_BBOX[2])),
                            "lon": float(rng.uniform(DHAKA_BBOX[1], DHAKA_BBOX[3])),
                        }
                    ],
                }
            )

        with (raw_dir / "dhaka_osm_features_fallback.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump({"elements": fallback}, file, ensure_ascii=True, indent=2)
        return fallback


def _centroid(geometry: list[dict[str, float]]) -> tuple[float, float]:
    lats = [float(p["lat"]) for p in geometry if "lat" in p and "lon" in p]
    lons = [float(p["lon"]) for p in geometry if "lat" in p and "lon" in p]
    return float(np.mean(lats)), float(np.mean(lons))


def _line_midpoint(path: list[list[float]]) -> tuple[float, float]:
    if not path:
        return 0.0, 0.0
    lat = sum(pt[0] for pt in path) / len(path)
    lon = sum(pt[1] for pt in path) / len(path)
    return float(lat), float(lon)


def _geometry_points(
    geometry: list[dict[str, float]],
    max_points: int = 25,
) -> list[tuple[float, float]]:
    valid = [(float(p["lat"]), float(p["lon"])) for p in geometry if "lat" in p and "lon" in p]
    if not valid:
        return []
    if len(valid) <= max_points:
        return valid
    step = max(1, len(valid) // max_points)
    sampled = valid[::step]
    if sampled[-1] != valid[-1]:
        sampled.append(valid[-1])
    return sampled


def _touched_wards_for_element(
    element: dict[str, Any],
    wards: list[WardPolygon],
) -> set[int]:
    geometry = element.get("geometry", [])
    if not geometry:
        return set()

    points = _geometry_points(geometry, max_points=25)
    if not points:
        return set()

    touched_wards: set[int] = set()
    for lat, lon in points:
        for ward in wards:
            if ward.contains(lat, lon):
                touched_wards.add(ward.ward_id)

    if not touched_wards:
        lat, lon = _centroid(geometry)
        for ward in wards:
            if ward.contains(lat, lon):
                touched_wards.add(ward.ward_id)
                break

    return touched_wards


def _sample_polyline(
    points: list[tuple[float, float]],
    max_points: int = 36,
) -> list[list[float]]:
    if len(points) <= 1:
        return []
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return [[round(lat, 6), round(lon, 6)] for lat, lon in points]


def _layer_bucket(tags: dict[str, Any]) -> str | None:
    if tags.get("highway"):
        return "roads"
    waterway = str(tags.get("waterway", "")).strip().lower()
    if waterway in {"river", "canal", "stream"}:
        return "rivers"
    if waterway:
        return "drains"
    if tags.get("natural") == "water":
        return "waterbodies"
    if tags.get("building"):
        return "houses"
    if tags.get("leisure") == "playground":
        return "playgrounds"
    if tags.get("leisure") == "park" or tags.get("landuse") == "grass":
        return "parks"
    return None


def _count_features_by_ward(
    elements: list[dict[str, Any]], wards: list[WardPolygon]
) -> dict[int, dict[str, int]]:
    counts = {
        ward.ward_id: {
            "waterway": 0,
            "highway": 0,
            "water": 0,
            "residential": 0,
            "green": 0,
            "house": 0,
            "playground": 0,
        }
        for ward in wards
    }

    for element in elements:
        tags = element.get("tags", {})
        feature = _feature_type(tags)
        if not feature:
            continue
        touched_wards = _touched_wards_for_element(element, wards)

        for ward_id in touched_wards:
            counts[ward_id][feature] += 1
    return counts


def _build_ward_map_layers(
    elements: list[dict[str, Any]],
    wards: list[WardPolygon],
) -> dict[int, dict[str, Any]]:
    layers = {
        ward.ward_id: {
            "roads": [],
            "drains": [],
            "rivers": [],
            "waterbodies": [],
            "houses": [],
            "playgrounds": [],
            "parks": [],
        }
        for ward in wards
    }
    limits = {
        "roads": 380,
        "drains": 220,
        "rivers": 180,
        "waterbodies": 140,
        "houses": 600,
        "playgrounds": 120,
        "parks": 180,
    }

    for element in elements:
        tags = element.get("tags", {})
        bucket = _layer_bucket(tags)
        geometry = element.get("geometry", [])
        if not bucket or not geometry:
            continue

        touched_wards = _touched_wards_for_element(element, wards)
        if not touched_wards:
            continue

        points = _geometry_points(geometry, max_points=40)
        if not points:
            continue

        polyline = _sample_polyline(points, max_points=38)
        element_id = int(element.get("id", 0))
        center_lat, center_lon = _centroid(geometry)

        for ward_id in touched_wards:
            ward_layer = layers[ward_id]
            if len(ward_layer[bucket]) >= limits[bucket]:
                continue

            if bucket in {"roads", "drains", "rivers", "waterbodies"}:
                if len(polyline) >= 2:
                    ward_layer[bucket].append(polyline)
                continue

            if bucket == "houses":
                height = 2.3 + (element_id % 10) * 0.45
                footprint = max(_polygon_area_km2(_ensure_closed(polyline)), 0.0005)
                ward_layer["houses"].append(
                    {
                        "lat": round(center_lat, 6),
                        "lon": round(center_lon, 6),
                        "height": round(height, 2),
                        "footprint": round(float(footprint), 5),
                    }
                )
                continue

            area = max(_polygon_area_km2(_ensure_closed(polyline)), 0.001)
            ward_layer[bucket].append(
                {
                    "lat": round(center_lat, 6),
                    "lon": round(center_lon, 6),
                    "area_km2": round(float(area), 6),
                    "size": round(float(np.clip(area * 80.0, 0.01, 1.8)), 4),
                }
            )

    for ward_id, ward_layer in layers.items():
        ward_layer["summary"] = {
            "roads": len(ward_layer["roads"]),
            "drains": len(ward_layer["drains"]),
            "rivers": len(ward_layer["rivers"]),
            "waterbodies": len(ward_layer["waterbodies"]),
            "houses": len(ward_layer["houses"]),
            "playgrounds": len(ward_layer["playgrounds"]),
            "parks": len(ward_layer["parks"]),
        }

    return layers


def _generate_ward_tables(
    wards: list[WardPolygon],
    feature_counts: dict[int, dict[str, int]],
    map_layers: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ward_rows: list[dict[str, Any]] = []
    indicator_rows: list[dict[str, Any]] = []
    morphologies = {
        ward.ward_id: summarize_ward_morphology(
            ward.boundary,
            map_layers.get(ward.ward_id, {}),
            feature_counts.get(ward.ward_id, {}),
            city_bbox=DHAKA_BBOX,
        )
        for ward in wards
    }
    score_map = score_ward_indicators(morphologies)

    for ward in wards:
        morphology = morphologies[ward.ward_id]
        score = score_map[ward.ward_id]

        ward_rows.append(
            {
                "id": ward.ward_id,
                "code": ward.code,
                "name": ward.name,
                "area_km2": round(morphology.area_km2, 3),
                "population": morphology.population,
                "households": morphology.households,
                "bbox_json": json.dumps(ward.boundary),
            }
        )

        indicator_rows.append(
            {
                "ward_id": ward.ward_id,
                "informal_area_pct": score.informal_area_pct,
                "blocked_drain_count": score.blocked_drain_count,
                "green_deficit_index": score.green_deficit_index,
                "flood_risk_index": score.flood_risk_index,
                "sdg11_score": score.sdg11_score,
                "exposed_population": score.exposed_population,
            }
        )

    return pd.DataFrame(ward_rows), pd.DataFrame(indicator_rows)


def _generate_interventions(
    wards_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    feature_counts: dict[int, dict[str, int]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    categories = [
        ("Drain cleaning micro-package", "Drainage", "City Corporation", (0.35, 1.6), 0.08),
        ("Road patch and resurfacing", "Road", "LGED", (0.55, 2.8), 0.22),
        ("Waste hotspot clearance", "Waste", "City Corporation", (0.25, 1.2), 0.05),
        ("Pocket park and tree corridor", "Green", "RAJUK", (0.45, 2.2), 0.32),
        ("Canal edge desilting", "Water", "WASA", (0.8, 3.5), 0.4),
        ("Community safety lighting", "Public Safety", "City Corporation", (0.3, 1.4), 0.1),
    ]
    rows: list[dict[str, Any]] = []
    intervention_id = 1

    indicators_map = indicators_df.set_index("ward_id").to_dict(orient="index")

    def infer_beneficiaries(
        *,
        category: str,
        estimated_cost_lakh: float,
        population: int,
        households: int,
        flood_risk: float,
        blocked_drain_count: float,
        green_deficit: float,
        informal_area_pct: float,
        road_count: int,
        residential_count: int,
    ) -> tuple[int, int, int, float]:
        road_density = road_count / max(households, 1)
        residential_density = residential_count / max(households, 1)

        affected_ratio_mean = {
            "Drainage": 0.14 + flood_risk * 0.56 + blocked_drain_count / 420.0,
            "Water": 0.12 + flood_risk * 0.48 + blocked_drain_count / 520.0,
            "Road": 0.10 + road_density * 0.42 + flood_risk * 0.08,
            "Waste": 0.13 + residential_density * 0.8 + informal_area_pct / 260.0,
            "Green": 0.08 + green_deficit * 0.5 + informal_area_pct / 360.0,
            "Public Safety": 0.09 + informal_area_pct / 210.0 + road_density * 0.25,
        }.get(category, 0.1)
        affected_ratio_mean = float(np.clip(affected_ratio_mean, 0.04, 0.92))

        cost_scale = {
            "Drainage": 1.1,
            "Water": 1.45,
            "Road": 1.75,
            "Waste": 0.85,
            "Green": 1.25,
            "Public Safety": 0.95,
        }.get(category, 1.0)

        # Bayesian Monte Carlo inference:
        # 1) sample uncertain coverage and affected households
        # 2) update Gamma prior with pseudo-observation intensity
        # 3) draw posterior-predictive beneficiaries using NegBin overdispersion
        sample_n = 350
        uncertainty = float(np.clip(0.12 + flood_risk * 0.18 + informal_area_pct / 700.0, 0.08, 0.34))
        concentration = float(np.clip(38.0 - uncertainty * 60.0, 12.0, 34.0))
        alpha_ratio = max(affected_ratio_mean * concentration, 1.1)
        beta_ratio = max((1.0 - affected_ratio_mean) * concentration, 1.1)
        affected_samples = rng.beta(alpha_ratio, beta_ratio, size=sample_n)

        coverage_mean = 1.0 - float(np.exp(-estimated_cost_lakh / cost_scale))
        coverage_sd = float(np.clip(0.06 + estimated_cost_lakh * 0.01, 0.05, 0.2))
        coverage_samples = np.clip(rng.normal(coverage_mean, coverage_sd, size=sample_n), 0.05, 0.98)

        exposed_households = households * affected_samples * coverage_samples
        household_size = 4.6
        prior_alpha = {
            "Drainage": 3.8,
            "Water": 3.5,
            "Road": 2.9,
            "Waste": 3.1,
            "Green": 2.4,
            "Public Safety": 2.7,
        }.get(category, 3.0)
        prior_beta = {
            "Drainage": 1.2,
            "Water": 1.25,
            "Road": 1.45,
            "Waste": 1.3,
            "Green": 1.6,
            "Public Safety": 1.5,
        }.get(category, 1.35)
        evidence_strength = float(np.clip(0.55 + road_density * 6.0 + residential_density * 4.0, 0.6, 1.7))
        posterior_mean_samples = (
            prior_alpha + (exposed_households * household_size * evidence_strength)
        ) / (prior_beta + evidence_strength)
        dispersion = float(
            np.clip(
                10.0 - flood_risk * 2.8 + (0.8 if category in {"Drainage", "Water"} else 0.0),
                4.8,
                12.5,
            )
        )
        nb_prob = np.clip(dispersion / (dispersion + posterior_mean_samples), 0.01, 0.999)
        predictive = rng.negative_binomial(dispersion, nb_prob, size=sample_n)
        predictive = np.clip(predictive, 90, int(population * 0.88))

        expected = int(np.clip(round(float(np.mean(predictive))), 120, population * 0.82))
        ci_low = int(max(80, round(float(np.quantile(predictive, 0.12)))))
        ci_high = int(min(population, round(float(np.quantile(predictive, 0.88)))))
        impact_per_lakh = float(round(expected / max(estimated_cost_lakh, 0.15), 3))
        return expected, ci_low, ci_high, impact_per_lakh

    for ward_row in wards_df.to_dict(orient="records"):
        ward_id = int(ward_row["id"])
        pop = int(ward_row["population"])
        households = int(ward_row["households"])
        ward_indicator = indicators_map[ward_id]
        flood_risk = float(ward_indicator["flood_risk_index"])
        equity_bias = float(ward_indicator["informal_area_pct"]) / 100.0
        feat = feature_counts.get(ward_id, {})
        road_count = int(feat.get("highway", 0))
        residential_count = int(feat.get("residential", 0))
        blocked_drain_count = float(ward_indicator["blocked_drain_count"])
        green_deficit = float(ward_indicator["green_deficit_index"])
        informal_area_pct = float(ward_indicator["informal_area_pct"])

        for idx in range(34):
            label, category, agency, cost_band, permit_prob = categories[idx % len(categories)]
            cost = float(rng.uniform(*cost_band))
            beneficiaries, ci_low, ci_high, impact_per_lakh = infer_beneficiaries(
                category=category,
                estimated_cost_lakh=cost,
                population=pop,
                households=households,
                flood_risk=flood_risk,
                blocked_drain_count=blocked_drain_count,
                green_deficit=green_deficit,
                informal_area_pct=informal_area_pct,
                road_count=road_count,
                residential_count=residential_count,
            )
            feasibility = float(np.clip(rng.normal(0.74, 0.15), 0.2, 0.99))
            equity_need = float(np.clip(0.35 + equity_bias * 0.5 + rng.normal(0, 0.08), 0.05, 0.99))
            urgency = float(
                np.clip(
                    0.38
                    + flood_risk * 0.42
                    + (0.12 if category in {"Drainage", "Water"} else 0)
                    + rng.normal(0, 0.07),
                    0.05,
                    0.99,
                )
            )
            impact_signal = (
                (beneficiaries / (cost * 10_000))
                + feasibility * 0.22
                + equity_need * 0.28
                + urgency * 0.33
                + rng.normal(0, 0.02)
            )

            rows.append(
                {
                    "id": intervention_id,
                    "ward_id": ward_id,
                    "title": f"{label} - Cluster {idx + 1}",
                    "category": category,
                    "agency": agency,
                    "permit_required": rng.random() < permit_prob,
                    "estimated_cost_lakh": round(cost, 3),
                    "expected_beneficiaries": beneficiaries,
                    "beneficiary_ci_low": ci_low,
                    "beneficiary_ci_high": ci_high,
                    "beneficiary_method": "bayesian-gamma-poisson-montecarlo-v2",
                    "impact_per_lakh": impact_per_lakh,
                    "feasibility": round(feasibility, 4),
                    "equity_need": round(equity_need, 4),
                    "urgency": round(urgency, 4),
                    "impact_signal": float(round(impact_signal, 5)),
                }
            )
            intervention_id += 1
    return pd.DataFrame(rows)


def _nearby_house_density(
    houses: list[dict[str, Any]],
    lat: float,
    lon: float,
    radius: float = 0.0026,
) -> float:
    count = 0
    for house in houses:
        h_lat = float(house.get("lat", 0.0))
        h_lon = float(house.get("lon", 0.0))
        if abs(h_lat - lat) <= radius and abs(h_lon - lon) <= radius:
            count += 1
    return float(count)


def _generate_monitoring_training_data(
    wards_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    map_layers: dict[int, dict[str, Any]],
    feature_counts: dict[int, dict[str, int]],
    ward_boundaries: dict[int, list[list[float]]],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators_map = indicators_df.set_index("ward_id").to_dict(orient="index")

    drain_rows: list[dict[str, Any]] = []
    flood_rows: list[dict[str, Any]] = []

    for ward_row in wards_df.to_dict(orient="records"):
        ward_id = int(ward_row["id"])
        ward_layers = map_layers.get(ward_id, {})
        drains = ward_layers.get("drains", []) if isinstance(ward_layers, dict) else []
        waters = ward_layers.get("waterbodies", []) if isinstance(ward_layers, dict) else []
        rivers = ward_layers.get("rivers", []) if isinstance(ward_layers, dict) else []
        houses = ward_layers.get("houses", []) if isinstance(ward_layers, dict) else []
        indicator = indicators_map[ward_id]
        flood_risk = float(indicator["flood_risk_index"])
        green_deficit = float(indicator["green_deficit_index"])
        morphology = summarize_ward_morphology(
            ward_boundaries.get(ward_id, []),
            ward_layers if isinstance(ward_layers, dict) else {},
            feature_counts.get(ward_id, {}),
            city_bbox=DHAKA_BBOX,
        )
        blocked_ratio = float(
            np.clip(
                float(indicator["blocked_drain_count"]) / max(estimate_blocked_network_scale(morphology), 1.0),
                0.0,
                1.0,
            )
        )

        water_points = [
            _line_midpoint(path)
            for path in [*waters, *rivers]
            if isinstance(path, list) and len(path) >= 2
        ]
        if not water_points:
            water_points = [(23.8 + rng.normal(0, 0.01), 90.4 + rng.normal(0, 0.01))]

        edge_rows: list[dict[str, Any]] = []
        for path in drains[:220]:
            if not isinstance(path, list) or len(path) < 2:
                continue
            for edge_idx in range(len(path) - 1):
                p1 = path[edge_idx]
                p2 = path[edge_idx + 1]
                if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) == 2 and len(p2) == 2):
                    continue
                mid_lat = float((p1[0] + p2[0]) / 2.0)
                mid_lon = float((p1[1] + p2[1]) / 2.0)
                near_water_dist = float(
                    min(abs(mid_lat - w_lat) + abs(mid_lon - w_lon) for w_lat, w_lon in water_points)
                )
                water_proximity = float(np.clip(1.0 - near_water_dist / 0.06, 0.0, 1.0))
                house_density = _nearby_house_density(houses, mid_lat, mid_lon, radius=0.0028)
                citizen_pressure = float(
                    np.clip(
                        rng.normal(0.2 + blocked_ratio * 0.35 + flood_risk * 0.22, 0.15),
                        0.0,
                        1.0,
                    )
                )
                rainfall_sensor_mm = float(np.clip(rng.normal(22 + flood_risk * 80, 12), 0.0, 130.0))
                pump_runtime_hours = float(np.clip(rng.normal(6.5 - flood_risk * 2.3, 1.7), 0.4, 12.0))
                last_maintenance_days = float(
                    np.clip(rng.normal(40 + blocked_ratio * 110, 24), 2.0, 260.0)
                )
                segment_length_m = float(
                    np.clip(
                        (((p2[0] - p1[0]) * 111_000) ** 2 + ((p2[1] - p1[1]) * 111_000) ** 2) ** 0.5,
                        4.0,
                        280.0,
                    )
                )

                latent = (
                    -1.45
                    + water_proximity * 1.75
                    + citizen_pressure * 1.25
                    + (last_maintenance_days / 180.0) * 0.95
                    + (rainfall_sensor_mm / 120.0) * 0.65
                    + (house_density / 30.0) * 0.5
                    - pump_runtime_hours * 0.08
                    + rng.normal(0, 0.35)
                )
                blocked_target = 1 if latent > 0 else 0
                edge_rows.append(
                    {
                        "ward_id": ward_id,
                        "segment_length_m": round(segment_length_m, 3),
                        "water_proximity": round(water_proximity, 5),
                        "house_density": round(house_density, 4),
                        "citizen_pressure": round(citizen_pressure, 5),
                        "rainfall_sensor_mm": round(rainfall_sensor_mm, 4),
                        "pump_runtime_hours": round(pump_runtime_hours, 4),
                        "last_maintenance_days": round(last_maintenance_days, 4),
                        "blocked_target": int(blocked_target),
                        "mid_lat": round(mid_lat, 6),
                        "mid_lon": round(mid_lon, 6),
                    }
                )

        if edge_rows:
            drain_rows.extend(edge_rows)
            drain_blocked_rate = float(sum(row["blocked_target"] for row in edge_rows) / len(edge_rows))
        else:
            drain_blocked_rate = float(np.clip(blocked_ratio, 0.02, 0.92))

        flood_candidates: list[tuple[float, float]] = []
        for path in [*waters[:90], *rivers[:90]]:
            if isinstance(path, list) and path:
                flood_candidates.append(_line_midpoint(path))
        for path in drains[:120]:
            if isinstance(path, list) and path:
                flood_candidates.append(_line_midpoint(path))
        if not flood_candidates:
            flood_candidates = [(23.8 + rng.normal(0, 0.015), 90.4 + rng.normal(0, 0.015))]

        for idx, (lat, lon) in enumerate(flood_candidates[:320]):
            near_water_dist = float(
                min(abs(lat - w_lat) + abs(lon - w_lon) for w_lat, w_lon in water_points)
            )
            water_proximity = float(np.clip(1.0 - near_water_dist / 0.085, 0.0, 1.0))
            impervious_surface = float(
                np.clip(
                    _nearby_house_density(houses, lat, lon, radius=0.0032) / 36.0
                    + min(morphology.house_footprint_km2 / max(morphology.area_km2, 0.02), 0.25) * 0.8,
                    0.0,
                    1.0,
                )
            )
            drainage_congestion = float(
                np.clip(
                    0.58 * blocked_ratio
                    + 0.26 * drain_blocked_rate
                    + rng.normal(0.0, 0.05)
                    + (idx % 5) * 0.01,
                    0.0,
                    1.0,
                )
            )
            lat_position = (lat - DHAKA_BBOX[0]) / max(DHAKA_BBOX[2] - DHAKA_BBOX[0], 1e-6)
            lon_position = (lon - DHAKA_BBOX[1]) / max(DHAKA_BBOX[3] - DHAKA_BBOX[1], 1e-6)
            lowland_signal = 0.62 * (1.0 - lat_position) + 0.38 * lon_position
            elevation_proxy = float(np.clip(0.88 - lowland_signal * 0.55 + rng.normal(0, 0.05), 0.05, 0.95))
            rainfall_sensor_mm = float(np.clip(rng.normal(12 + flood_risk * 74, 18), 0.0, 140.0))
            citizen_flood_pressure = float(
                np.clip(rng.normal(0.16 + flood_risk * 0.35 + blocked_ratio * 0.16, 0.14), 0.0, 1.0)
            )

            latent = (
                -3.1
                + water_proximity * 1.05
                + drainage_congestion * 1.00
                + impervious_surface * 0.82
                + (1.0 - elevation_proxy) * 0.80
                + (rainfall_sensor_mm / 140.0) * 0.84
                + citizen_flood_pressure * 0.72
                + flood_risk * 1.15
                + green_deficit * 0.52
                + rng.normal(0.0, 0.22)
            )
            target = float(np.clip(1.0 / (1.0 + np.exp(-latent)), 0.01, 0.99))
            flood_rows.append(
                {
                    "ward_id": ward_id,
                    "water_proximity": round(water_proximity, 5),
                    "drainage_congestion": round(drainage_congestion, 5),
                    "impervious_surface": round(impervious_surface, 5),
                    "elevation_proxy": round(elevation_proxy, 5),
                    "rainfall_sensor_mm": round(rainfall_sensor_mm, 4),
                    "citizen_flood_pressure": round(citizen_flood_pressure, 5),
                    "target_risk": round(target, 6),
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                }
            )

    drain_df = pd.DataFrame(drain_rows)
    if not drain_df.empty:
        positives = int(drain_df["blocked_target"].sum())
        min_positive = max(20, int(len(drain_df) * 0.08))
        if positives < min_positive:
            deficit = min_positive - positives
            candidate_idx = drain_df[drain_df["blocked_target"] == 0].index.to_numpy()
            if len(candidate_idx) > 0:
                flip = rng.choice(candidate_idx, size=min(deficit, len(candidate_idx)), replace=False)
                drain_df.loc[flip, "blocked_target"] = 1
    flood_df = pd.DataFrame(flood_rows)
    return drain_df, flood_df


def _generate_civic_report_training(
    ward_ids: list[int], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    templates = {
        "blocked_drain": [
            "ড্রেন বন্ধ হয়ে আছে, পানি নামছে না",
            "Drain near lane {lane} is clogged and overflowing",
            "বর্ষায় রাস্তার পাশের নালা উপচে পড়ে",
            "Need urgent drain cleaning in block {lane}",
            "পচা আবর্জনায় ড্রেন আটকে গেছে",
        ],
        "flooding": [
            "Heavy rain causes waterlogging every evening",
            "আমাদের এলাকায় বৃষ্টির পরে হাঁটু পানি থাকে",
            "Road remains flooded for hours after rainfall",
            "স্কুলের সামনে জলাবদ্ধতা খুব বেশি",
            "Frequent flood around ward market",
        ],
        "waste": [
            "Garbage pile has not been removed for three days",
            "ডাস্টবিন না থাকায় রাস্তায় ময়লা জমছে",
            "Waste hotspot beside the canal is growing",
            "মশার উপদ্রব হচ্ছে ময়লার কারণে",
            "Need regular waste collection in sector {lane}",
        ],
        "road_damage": [
            "রাস্তার গর্তে যানজট ও দুর্ঘটনা হচ্ছে",
            "Potholes near junction {lane} damage vehicles",
            "Road patch repair is needed urgently",
            "Ambulance cannot pass because road is broken",
            "পাকা রাস্তা উঠে গেছে, দ্রুত মেরামত দরকার",
        ],
        "water_supply": [
            "Water pressure is too low in the morning",
            "পানির লাইনে সমস্যা, বিশুদ্ধ পানি পাওয়া যাচ্ছে না",
            "Need WASA support for pipeline leak",
            "Drinking water smells bad this week",
            "নতুন সংযোগের আবেদন করেছি, এখনো কাজ হয়নি",
        ],
    }

    train_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for category, lines in templates.items():
        for _ in range(75):
            text = rng.choice(lines).format(lane=int(rng.integers(1, 22)))
            language = "bangla" if any("\u0980" <= char <= "\u09ff" for char in text) else "english"
            sentiment = float(np.clip(rng.normal(-0.65, 0.2), -1.0, 0.1))
            priority = float(
                np.clip(
                    0.45
                    + (0.25 if category in {"blocked_drain", "flooding"} else 0.1)
                    + rng.normal(0, 0.09),
                    0.15,
                    1.0,
                )
            )
            train_rows.append(
                {
                    "text": text,
                    "language": language,
                    "category": category,
                    "sentiment_score": round(sentiment, 4),
                    "priority_weight": round(priority, 4),
                }
            )

    seed_count = max(25, len(ward_ids) * 4)
    for _ in range(seed_count):
        category = rng.choice(list(templates.keys()))
        text = rng.choice(templates[category]).format(lane=int(rng.integers(1, 20)))
        ward_id = int(rng.choice(ward_ids))
        seed_rows.append(
            {
                "ward_id": ward_id,
                "text": text,
                "language": "bangla" if any("\u0980" <= char <= "\u09ff" for char in text) else "english",
                "category": category,
                "sentiment_score": float(np.clip(rng.normal(-0.5, 0.2), -1.0, 0.2)),
                "priority_weight": float(np.clip(rng.normal(0.6, 0.2), 0.1, 1.0)),
            }
        )

    return pd.DataFrame(train_rows), pd.DataFrame(seed_rows)


def _draw_blob(mask: np.ndarray, rng: np.random.Generator) -> None:
    h, w = mask.shape
    cx = rng.uniform(0.15 * w, 0.85 * w)
    cy = rng.uniform(0.15 * h, 0.85 * h)
    rx = rng.uniform(5, 18)
    ry = rng.uniform(5, 16)

    yy, xx = np.indices(mask.shape)
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    mask[ellipse] = 1.0


def _generate_segmentation_data(
    indicators_df: pd.DataFrame, rng: np.random.Generator, out_path: Path
) -> None:
    image_size = 64
    sample_count = 280
    images = np.zeros((sample_count, 1, image_size, image_size), dtype=np.float32)
    masks = np.zeros((sample_count, 1, image_size, image_size), dtype=np.float32)

    for idx in range(sample_count):
        base = rng.normal(0.35, 0.08, size=(image_size, image_size)).astype(np.float32)
        mask = np.zeros((image_size, image_size), dtype=np.float32)
        blob_count = int(rng.integers(1, 5))
        for _ in range(blob_count):
            _draw_blob(mask, rng)
        texture = rng.normal(0.15, 0.05, size=(image_size, image_size)).astype(np.float32)
        image = np.clip(base + mask * 0.45 + texture, 0.0, 1.0)
        images[idx, 0] = image
        masks[idx, 0] = mask

    ward_images = np.zeros((len(indicators_df), 1, image_size, image_size), dtype=np.float32)
    ward_ids = indicators_df["ward_id"].to_numpy(dtype=np.int32)
    for i, row in enumerate(indicators_df.to_dict(orient="records")):
        target_ratio = float(row["informal_area_pct"]) / 100.0
        image = rng.normal(0.32, 0.1, size=(image_size, image_size)).astype(np.float32)
        mask = np.zeros((image_size, image_size), dtype=np.float32)
        attempts = 0
        while mask.mean() < target_ratio and attempts < 8:
            _draw_blob(mask, rng)
            attempts += 1
        image = np.clip(image + mask * 0.42 + rng.normal(0.08, 0.04, image.shape), 0.0, 1.0)
        ward_images[i, 0] = image

    np.savez_compressed(
        out_path,
        train_images=images,
        train_masks=masks,
        ward_images=ward_images,
        ward_ids=ward_ids,
    )


def prepare_datasets(processed_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    wards, boundary_source = _download_ward_boundaries(raw_dir)
    osm_elements = _download_osm_features(raw_dir, rng)
    feature_counts = _count_features_by_ward(osm_elements, wards)
    map_layers = _build_ward_map_layers(osm_elements, wards)

    wards_df, indicators_df = _generate_ward_tables(wards, feature_counts, map_layers)
    interventions_df = _generate_interventions(wards_df, indicators_df, feature_counts, rng)
    drain_train_df, flood_train_df = _generate_monitoring_training_data(
        wards_df=wards_df,
        indicators_df=indicators_df,
        map_layers=map_layers,
        feature_counts=feature_counts,
        ward_boundaries={ward.ward_id: ward.boundary for ward in wards},
        rng=rng,
    )
    ward_ids = wards_df["id"].astype(int).tolist()
    report_train_df, seed_reports_df = _generate_civic_report_training(ward_ids, rng)

    wards_df.to_csv(processed_dir / "wards.csv", index=False)
    indicators_df.to_csv(processed_dir / "ward_indicators.csv", index=False)
    interventions_df.to_csv(processed_dir / "interventions.csv", index=False)
    drain_train_df.to_csv(processed_dir / "drainage_monitor_train.csv", index=False)
    flood_train_df.to_csv(processed_dir / "flood_risk_train.csv", index=False)
    report_train_df.to_csv(processed_dir / "civic_reports_train.csv", index=False)
    seed_reports_df.to_csv(processed_dir / "seed_reports.csv", index=False)

    with (processed_dir / "ward_feature_counts.json").open("w", encoding="utf-8") as file:
        json.dump(feature_counts, file, ensure_ascii=True, indent=2)

    with (processed_dir / "ward_boundaries.json").open("w", encoding="utf-8") as file:
        json.dump({ward.ward_id: ward.boundary for ward in wards}, file, ensure_ascii=True, indent=2)

    with (processed_dir / "ward_map_layers.json").open("w", encoding="utf-8") as file:
        json.dump({str(ward_id): payload for ward_id, payload in map_layers.items()}, file, ensure_ascii=True, indent=2)

    _generate_segmentation_data(indicators_df, rng, processed_dir / "segmentation_data.npz")

    layer_totals = {
        key: int(
            sum(
                ward_layer["summary"].get(key, 0)
                for ward_layer in map_layers.values()
            )
        )
        for key in ["roads", "drains", "rivers", "waterbodies", "houses", "playgrounds", "parks"]
    }

    meta = {
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_source": boundary_source,
        "map_source": "openstreetmap_overpass_api",
        "real_data_only": REAL_DATA_ONLY,
        "ward_count": int(len(wards)),
        "feature_total": int(len(osm_elements)),
        "layer_totals": layer_totals,
        "drain_training_rows": int(len(drain_train_df)),
        "flood_training_rows": int(len(flood_train_df)),
    }
    with (processed_dir / "dataset_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=True, indent=2)
