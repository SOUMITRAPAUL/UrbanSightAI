from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE_TTL = timedelta(minutes=10)
_weather_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 3)}:{round(lon, 3)}"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_open_meteo_current(lat: float, lon: float, timeout: int = 12) -> dict[str, Any]:
    key = _cache_key(lat, lon)
    now = datetime.now(timezone.utc)
    cached = _weather_cache.get(key)
    if cached and now - cached[0] <= _CACHE_TTL:
        return dict(cached[1])

    params = {
        "latitude": round(float(lat), 5),
        "longitude": round(float(lon), 5),
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
            ]
        ),
        "hourly": "precipitation_probability",
        "forecast_days": 1,
        "timezone": "UTC",
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {})
        hourly = payload.get("hourly", {})
        hourly_prob = hourly.get("precipitation_probability", [])

        result = {
            "source": "open-meteo",
            "status": "ok",
            "observed_at": _parse_time(current.get("time")),
            "temperature_c": (
                float(current.get("temperature_2m"))
                if current.get("temperature_2m") is not None
                else None
            ),
            "relative_humidity_pct": (
                float(current.get("relative_humidity_2m"))
                if current.get("relative_humidity_2m") is not None
                else None
            ),
            "precipitation_mm": (
                float(current.get("precipitation"))
                if current.get("precipitation") is not None
                else None
            ),
            "rain_mm": (
                float(current.get("rain"))
                if current.get("rain") is not None
                else None
            ),
            "wind_speed_kmh": (
                float(current.get("wind_speed_10m"))
                if current.get("wind_speed_10m") is not None
                else None
            ),
            "precipitation_probability_pct": (
                float(hourly_prob[0]) if isinstance(hourly_prob, list) and hourly_prob else None
            ),
        }
        _weather_cache[key] = (now, result)
        return dict(result)
    except Exception:
        if cached:
            stale = dict(cached[1])
            stale["status"] = "stale_cache"
            return stale
        return {
            "source": "open-meteo",
            "status": "unavailable",
            "observed_at": None,
            "temperature_c": None,
            "relative_humidity_pct": None,
            "precipitation_mm": None,
            "rain_mm": None,
            "wind_speed_kmh": None,
            "precipitation_probability_pct": None,
        }
