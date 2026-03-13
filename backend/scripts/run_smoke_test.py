from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text

        login = client.post(
            "/api/auth/login",
            json={"username": "planner", "password": "pilot123"},
        )
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        wards = client.get("/api/wards", headers=headers)
        assert wards.status_code == 200, wards.text
        ward_id = wards.json()[0]["id"]

        twin = client.get(f"/api/wards/{ward_id}/digital-twin", headers=headers)
        assert twin.status_code == 200, twin.text

        scene = client.get(
            f"/api/wards/{ward_id}/digital-twin-scene", headers=headers
        )
        assert scene.status_code == 200, scene.text
        scene_body = scene.json()
        assert "blocked_drain_network" in scene_body["layers"]
        summary = scene_body["layers"]["summary"]
        for key in [
            "roads",
            "drains",
            "rivers",
            "waterbodies",
            "houses",
            "playgrounds",
            "parks",
            "blocked_drain_network",
            "blocked_drains",
            "flood_zones",
            "informal_zones",
        ]:
            assert summary[key] == len(scene_body["layers"][key])
        if scene_body["scores"]["blocked_drain_count"] > 0:
            assert summary["drains"] > 0
            assert summary["blocked_drain_network"] > 0
            assert scene_body["scores"]["blocked_drain_count"] == summary["blocked_drain_network"]

        workflow = client.get(
            f"/api/wards/{ward_id}/workflow", headers=headers
        )
        assert workflow.status_code == 200, workflow.text
        workflow_body = workflow.json()
        assert len(workflow_body["stages"]) == 7
        assert len(workflow_body["candidate_micro_works"]) >= 3

        card = client.get(
            f"/api/wards/{ward_id}/sdg11-card", headers=headers
        )
        assert card.status_code == 200, card.text

        packet = client.get(
            f"/api/wards/{ward_id}/interagency-packet", headers=headers
        )
        assert packet.status_code == 200, packet.text

        components = client.get("/api/ai-components", headers=headers)
        assert components.status_code == 200, components.text

        audit = client.get(f"/api/audit-trail?ward_id={ward_id}&limit=10", headers=headers)
        assert audit.status_code == 200, audit.text

        worklist = client.get(
            f"/api/wards/{ward_id}/top-worklist?top_n=10", headers=headers
        )
        assert worklist.status_code == 200, worklist.text

        scenario = client.get(
            f"/api/wards/{ward_id}/scenario?budget_lakh=9.2&strategy=climate_resilience",
            headers=headers,
        )
        assert scenario.status_code == 200, scenario.text
        scenario_body = scenario.json()
        assert "decision_basis" in scenario_body
        assert "selected_reasoning" in scenario_body
        assert scenario_body["strategy_profile"] == "climate_resilience"
        assert scenario_body["portfolio_summary"]["selected_count"] >= 1
        assert len(scenario_body["implementation_roadmap"]) >= 1
        assert len(scenario_body["strategy_comparison"]) == 4

        classify = client.post(
            "/api/reports/classify",
            headers=headers,
            json={
                "ward_id": ward_id,
                "text": "ড্রেন বন্ধ হয়ে আছে, পানি জমে আছে",
                "language": "bangla",
            },
        )
        assert classify.status_code == 200, classify.text

        report = client.post(
            "/api/reports",
            headers=headers,
            json={
                "ward_id": ward_id,
                "text": "Drain is blocked near school gate and water is overflowing",
                "language": "english",
            },
        )
        assert report.status_code == 200, report.text

        odk = client.post(
            "/api/reports/odk-submit",
            headers=headers,
            json={
                "ward_id": ward_id,
                "text": "Field app report: flood near lane 5",
                "language": "english",
                "issue_type": "flooding",
                "severity": "high",
                "location_hint": "Lane-5",
                "reporter_name": "enum-smoke",
            },
        )
        assert odk.status_code == 200, odk.text

        live = client.post(
            f"/api/wards/{ward_id}/live-ingest",
            headers=headers,
            json={"citizen_reports": 2, "odk_forms": 1, "sensor_pulses": 1},
        )
        assert live.status_code == 200, live.text

        notifications = client.get(
            f"/api/wards/{ward_id}/notification-feed?limit=10",
            headers=headers,
        )
        assert notifications.status_code == 200, notifications.text

        csv_export = client.get(
            f"/api/exports/wards/{ward_id}/worklist.csv",
            headers=headers,
        )
        assert csv_export.status_code == 200, csv_export.text

        pdf_export = client.get(
            (
                f"/api/exports/wards/{ward_id}/policy-memo.pdf?"
                "budget_lakh=9.2&strategy=climate_resilience"
            ),
            headers=headers,
        )
        assert pdf_export.status_code == 200, pdf_export.text

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
