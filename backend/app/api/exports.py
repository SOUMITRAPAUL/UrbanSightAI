from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dashboard import run_scenario, top_worklist
from app.api.deps import require_roles
from app.core.config import EXPORT_DIR
from app.db import get_db
from app.models import CitizenReport, User, Ward, WardIndicator
from app.schemas import PlanningStrategy
from app.services.audit import log_audit_event
from app.services.policy_export import export_policy_pdf, export_worklist_csv


router = APIRouter(prefix="/api/exports", tags=["exports"])


@router.get("/wards/{ward_id}/worklist.csv")
def export_worklist(
    ward_id: int,
    top_n: int = Query(default=10, ge=3, le=30),
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("planner", "enumerator")),
) -> FileResponse:
    worklist = top_worklist(ward_id=ward_id, top_n=top_n, db=db, _=user, _emit_audit=False)
    output_path = EXPORT_DIR / f"ward_{ward_id}_top_worklist.csv"
    export_worklist_csv(output_path, [item.model_dump() for item in worklist.items])
    log_audit_event(
        db,
        action="export_worklist_csv",
        user=user,
        ward_id=ward_id,
        details={"top_n": top_n, "filename": output_path.name},
    )
    return FileResponse(path=output_path, filename=output_path.name, media_type="text/csv")


@router.get("/wards/{ward_id}/policy-memo.pdf")
def export_policy_memo(
    ward_id: int,
    budget_lakh: float = Query(default=8.0, gt=0.4, le=120.0),
    strategy: PlanningStrategy = Query(default="balanced"),
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("planner")),
) -> FileResponse:
    # 1. Standard Scenario (Selection & Roadmap)
    scenario = run_scenario(
        ward_id=ward_id,
        budget_lakh=budget_lakh,
        strategy=strategy,
        db=db,
        _=user,
        _emit_audit=False,
    )
    
    # 2. AI Budget Plan (Sector Allocation)
    from app.api.dashboard import ai_budget_plan
    ai_plan = ai_budget_plan(ward_id=ward_id, budget_lakh=budget_lakh, db=db, _=user)
    
    # 3. Indicators & Problems (Risk Profile)
    # We pull indicators directly from DB for the gauge snapshot
    ward = db.scalar(select(Ward).where(Ward.id == ward_id))
    ward_name = ward.name if ward else f"Ward {ward_id}"
    indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == ward_id))
    
    # Pull recent reports for problem context (similar to dashboard logic)
    report_rows = db.scalars(
        select(CitizenReport)
        .where(CitizenReport.ward_id == ward_id)
        .order_by(CitizenReport.created_at.desc())
        .limit(100)
    ).all()
    
    # Build problem indicators (we'll pass these to the service)
    indicators_snapshot = {
        "flood_risk": float(indicator.flood_risk_index) if indicator else 0.0,
        "blocked_drains": int(indicator.blocked_drain_count) if indicator else 0,
        "informal_area": float(indicator.informal_area_pct) if indicator else 0.0,
        "exposed_pop": int(indicator.exposed_population) if indicator else 0,
        "green_deficit": float(indicator.green_deficit_index) if indicator else 0.0,
        "sdg11_score": float(indicator.sdg11_score) if indicator else 0.0,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    output_path = EXPORT_DIR / f"policy_memo_ward_{ward_id}_{stamp}.pdf"
    
    export_policy_pdf(
        output_path=output_path,
        ward_name=ward_name,
        budget_lakh=budget_lakh,
        strategy_label=scenario.strategy_label,
        strategy_description=scenario.strategy_description,
        selected_items=[item.model_dump() for item in scenario.selected_projects],
        counterfactuals=scenario.counterfactuals,
        portfolio_summary=scenario.portfolio_summary,
        roadmap=[item.model_dump() for item in scenario.implementation_roadmap],
        tradeoff_alerts=[item.model_dump() for item in scenario.tradeoff_alerts],
        ai_budget_plan=ai_plan,
        indicators=indicators_snapshot,
    )
    
    log_audit_event(
        db,
        action="export_policy_memo_pdf",
        user=user,
        ward_id=ward_id,
        details={
            "budget_lakh": budget_lakh,
            "strategy_profile": strategy,
            "filename": output_path.name,
        },
    )
    return FileResponse(path=output_path, filename=output_path.name, media_type="application/pdf")
