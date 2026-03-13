# UrbanSightAI Pilot Prototype

This repository implements the PDF proposal pilot using:

- **Frontend:** React (Vite)
- **Backend:** FastAPI + SQLite
- **ML components:** prioritization ranker, segmentation evidence extractor (Mini U-Net), civic NLP classifier, and policy scenario simulator

## What is implemented (mapped to proposal deliverables)

1. **AI Policy Prioritization Output**
- Top-N intervention ranking by impact/cost, feasibility, equity, urgency
- API: `GET /api/wards/{ward_id}/top-worklist`

2. **AI Governance Chatbot / Civic Intelligence**
- Bangla/English citizen report classification and priority scoring
- APIs: `POST /api/reports/classify`, `POST /api/reports`

3. **Ward-Level Digital Twin Dashboard**
- Interactive 3D digital twin (Three.js) with roads, drains, houses, playgrounds, parks, waterbodies
- Problem hotspots, scores, and action pipeline overlay
- APIs: `GET /api/wards/{ward_id}/digital-twin`, `GET /api/wards/{ward_id}/digital-twin-scene`

4. **Automated SDG-11 Governance Cards + Inter-Agency Packets + Audit Trail**
- SDG-11 target cards for ward-level governance reporting
- Inter-agency implementation packet/checklist for municipal coordination
- Queryable action audit logs for transparency and donor reporting
- APIs: `GET /api/wards/{ward_id}/sdg11-card`, `GET /api/wards/{ward_id}/interagency-packet`, `GET /api/audit-trail`

5. **Budget Scenario & Policy Recommendation Engine**
- Budget-to-impact counterfactual simulation with selected micro-works
- API: `GET /api/wards/{ward_id}/scenario`

6. **Integrated Web Platform**
- Role-based login/register (`planner`, `enumerator`, `viewer`)
- CSV and PDF export for policy packets
- APIs: `GET /api/exports/wards/{ward_id}/worklist.csv`, `GET /api/exports/wards/{ward_id}/policy-memo.pdf`

7. **Five AI Components Status**
- Tracks all five proposal AI components and output routes in one endpoint
- API: `GET /api/ai-components`

## Data strategy

- **Downloaded real data:** OSM urban features for Dhaka via Overpass API (saved in `backend/data/raw/`).
- **3D scene layers:** generated per ward from OSM geometries (roads, drains, houses, playgrounds, parks, waterbodies).
- **Digital twin boundaries:** real OSM polygons are used first:
  - admin ward boundaries (`admin_level=10`) if enough are available,
  - otherwise OSM place polygons (`suburb/neighbourhood/quarter`),
  - otherwise synthetic grid fallback.
- **Data-faithful indicators:** ward metrics are computed deterministically from observed OSM features (no random indicator noise).
- **Resilient caching:** if API download fails, generator reuses cached OSM boundary/feature files before synthetic fallback.
- **Synthetic data generation:** interventions, civic report training set, and segmentation imagery are generated for pilot robustness and fast reproducibility.
- If download fails, pipeline automatically falls back to synthetic OSM-like features.

## Run backend

```bash
cd backend
python3 -m venv --system-site-packages .venv_sys
source .venv_sys/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Default users:
- `planner / pilot123`
- `enumerator / pilot123`
- `viewer / pilot123`

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend expects backend at `http://127.0.0.1:8000` by default.

## Validation already run

- Backend smoke test:
```bash
cd backend
source .venv_sys/bin/activate
python scripts/run_smoke_test.py
```
Result: **passed**

- Backend test suite:
```bash
cd backend
source .venv_sys/bin/activate
pytest -q
```
Result: **1 passed**

- Frontend checks:
```bash
cd frontend
npm run lint
npm run build
```
Result: **passed**
