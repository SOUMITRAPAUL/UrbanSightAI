# UrbanSightAI Backend (FastAPI)

Pilot backend implementing:

- Role-based auth (`planner`, `enumerator`, `viewer`)
- Ward digital twin API
- 3D scene API for map layers + problem hotspots + action pipeline
- SDG-11 governance card API
- Inter-agency packet API
- Audit trail API
- Five AI components status API
- AI prioritization engine (Top-N micro-work list)
- Real-world constrained scenario simulator (budget reserve, OPEX cap, permit-share, agency capacity, timeline)
- Civic report classifier (Bangla/English complaint intake)
- Model trust and model-card APIs (`/api/model-trust`, `/api/model-cards`)
- CSV/PDF export endpoints for policy packets

## Run

```bash
cd backend
python3 -m venv --system-site-packages .venv_sys
source .venv_sys/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Default users

- `planner` / `pilot123`
- `enumerator` / `pilot123`
- `viewer` / `pilot123`

## Smoke test

```bash
cd backend
source .venv_sys/bin/activate
python scripts/run_smoke_test.py
```
