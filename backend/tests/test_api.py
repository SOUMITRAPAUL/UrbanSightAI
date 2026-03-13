from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_core_flow() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        login = client.post(
            "/api/auth/login",
            json={"username": "planner", "password": "pilot123"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        wards = client.get("/api/wards", headers=headers)
        assert wards.status_code == 200
        ward_id = wards.json()[0]["id"]

        scene = client.get(f"/api/wards/{ward_id}/digital-twin-scene", headers=headers)
        assert scene.status_code == 200
        scene_body = scene.json()
        assert "layers" in scene_body
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

        workflow = client.get(f"/api/wards/{ward_id}/workflow", headers=headers)
        assert workflow.status_code == 200
        workflow_body = workflow.json()
        assert len(workflow_body["stages"]) == 7
        assert len(workflow_body["candidate_micro_works"]) >= 3

        card = client.get(f"/api/wards/{ward_id}/sdg11-card", headers=headers)
        assert card.status_code == 200
        assert "targets" in card.json()

        packet = client.get(f"/api/wards/{ward_id}/interagency-packet", headers=headers)
        assert packet.status_code == 200
        assert "tasks" in packet.json()

        components = client.get("/api/ai-components", headers=headers)
        assert components.status_code == 200
        assert len(components.json()) == 5

        audit = client.get(f"/api/audit-trail?ward_id={ward_id}&limit=10", headers=headers)
        assert audit.status_code == 200

        top = client.get(f"/api/wards/{ward_id}/top-worklist", headers=headers)
        assert top.status_code == 200
        assert len(top.json()["items"]) >= 3

        scenario = client.get(
            f"/api/wards/{ward_id}/scenario?budget_lakh=9.2&strategy=equity_first",
            headers=headers,
        )
        assert scenario.status_code == 200
        scenario_body = scenario.json()
        assert "selection_method" in scenario_body
        assert "decision_basis" in scenario_body
        assert "selected_reasoning" in scenario_body
        assert scenario_body["strategy_profile"] == "equity_first"
        assert scenario_body["strategy_label"]
        assert "portfolio_summary" in scenario_body
        assert "agency_load" in scenario_body
        assert "implementation_roadmap" in scenario_body
        assert "tradeoff_alerts" in scenario_body
        assert "deferred_projects" in scenario_body
        assert "strategy_comparison" in scenario_body
        assert len(scenario_body["strategy_comparison"]) == 4
        assert any(
            item["strategy"] == "equity_first"
            for item in scenario_body["strategy_comparison"]
        )

        odk = client.post(
            "/api/reports/odk-submit",
            headers=headers,
            json={
                "ward_id": ward_id,
                "text": "Field form: drain blocked near school",
                "language": "english",
                "issue_type": "blocked_drain",
                "severity": "high",
                "location_hint": "Lane-3",
                "reporter_name": "enum-a",
            },
        )
        assert odk.status_code == 200

        live = client.post(
            f"/api/wards/{ward_id}/live-ingest",
            headers=headers,
            json={"citizen_reports": 2, "odk_forms": 1, "sensor_pulses": 1},
        )
        assert live.status_code == 200
        live_body = live.json()
        assert "updated_indicators" in live_body
        assert len(live_body["sources"]) >= 1

        notifications = client.get(
            f"/api/wards/{ward_id}/notification-feed?limit=10",
            headers=headers,
        )
        assert notifications.status_code == 200
