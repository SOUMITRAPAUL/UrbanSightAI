from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_roles
from app.db import get_db
from app.models import CitizenReport, User
from app.schemas import (
    CitizenReportPrediction,
    CitizenReportRequest,
    CitizenReportResponse,
    ODKSubmissionRequest,
    ChatRequest,
    ChatResponse,
)
from app.services.model_hub import MODEL_HUB
from app.services.audit import log_audit_event
from app.services.rag_service import RAG_SERVICE


router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    ward_context = None
    if payload.ward_id:
        from app.models import WardIndicator, Ward
        from app.api.dashboard import ai_budget_plan
        
        ward = db.scalar(select(Ward).where(Ward.id == payload.ward_id))
        indicator = db.scalar(select(WardIndicator).where(WardIndicator.ward_id == payload.ward_id))
        
        if indicator:
            ward_context = (
                f"WARD NAME: {ward.name if ward else 'Unknown'}\n"
                f"INDICATORS: Flood Risk: {indicator.flood_risk_index:.2f}, "
                f"Blocked Drains: {indicator.blocked_drain_count}, "
                f"Green Deficit: {indicator.green_deficit_index:.3f}, "
                f"Informal Area: {indicator.informal_area_pct:.1f}%, "
                f"Exposed Pop: {indicator.exposed_population}, "
                f"SDG-11 Score: {indicator.sdg11_score:.2f}"
            )
            
            # Add latest AI Budget Plan if available (simulating a default 8.0L budget for context)
            try:
                plan = ai_budget_plan(ward_id=payload.ward_id, budget_lakh=10.0, db=db, _=user)
                if plan:
                    ward_context += f"\nAI BUDGET PREDICTION (for 10L): {plan['ai_summary']}"
            except:
                pass

    response = await RAG_SERVICE.chat(payload.query, ward_context=ward_context)
    log_audit_event(
        db,
        action="chatbot_rag_query",
        user=user,
        ward_id=payload.ward_id,
        details={"query": payload.query, "intent": response.get("intent")},
    )
    return ChatResponse(**response)


@router.post("/classify", response_model=CitizenReportPrediction)
def classify_report(
    payload: CitizenReportRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CitizenReportPrediction:
    prediction = MODEL_HUB.classify_civic_report(payload.text)
    log_audit_event(
        db,
        action="civic_report_classified",
        user=user,
        ward_id=payload.ward_id,
        details={"category": prediction["category"], "confidence": prediction["confidence"]},
    )
    return CitizenReportPrediction(**prediction)


@router.post("", response_model=CitizenReportResponse)
def create_report(
    payload: CitizenReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("planner", "enumerator")),
) -> CitizenReportResponse:
    prediction = MODEL_HUB.classify_civic_report(payload.text)
    report = CitizenReport(
        ward_id=payload.ward_id,
        text=payload.text,
        language=payload.language,
        category=prediction["category"],
        sentiment_score=prediction["sentiment_score"],
        priority_weight=prediction["priority_weight"],
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    log_audit_event(
        db,
        action="civic_report_created",
        user=user,
        ward_id=payload.ward_id,
        details={
            "report_id": report.id,
            "category": report.category,
            "priority_weight": report.priority_weight,
        },
    )
    return CitizenReportResponse(
        id=report.id,
        ward_id=report.ward_id,
        text=report.text,
        language=report.language,
        category=report.category,
        sentiment_score=report.sentiment_score,
        priority_weight=report.priority_weight,
        created_at=report.created_at,
    )


@router.post("/odk-submit", response_model=CitizenReportResponse)
def submit_odk_form(
    payload: ODKSubmissionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("planner", "enumerator")),
) -> CitizenReportResponse:
    model_prediction = MODEL_HUB.classify_civic_report(payload.text)
    category = (
        model_prediction["category"]
        if payload.issue_type == "auto"
        else payload.issue_type
    )
    severity_weight = {
        "low": 0.08,
        "medium": 0.16,
        "high": 0.26,
        "critical": 0.34,
    }[payload.severity]
    priority = float(
        min(
            1.0,
            float(model_prediction["priority_weight"]) + severity_weight,
        )
    )
    location_tag = f"[ODK:{payload.location_hint}] " if payload.location_hint else "[ODK] "
    reporter_tag = f" (Field: {payload.reporter_name})" if payload.reporter_name else ""
    report = CitizenReport(
        ward_id=payload.ward_id,
        text=f"{location_tag}{payload.text}{reporter_tag}",
        language=payload.language,
        category=category,
        sentiment_score=float(model_prediction["sentiment_score"]),
        priority_weight=priority,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    log_audit_event(
        db,
        action="odk_submission_ingested",
        user=user,
        ward_id=payload.ward_id,
        details={
            "report_id": report.id,
            "source": "odk_kobo",
            "category": category,
            "severity": payload.severity,
            "location_hint": payload.location_hint,
        },
    )
    return CitizenReportResponse(
        id=report.id,
        ward_id=report.ward_id,
        text=report.text,
        language=report.language,
        category=report.category,
        sentiment_score=report.sentiment_score,
        priority_weight=report.priority_weight,
        created_at=report.created_at,
    )


@router.get("", response_model=list[CitizenReportResponse])
def list_reports(
    ward_id: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[CitizenReportResponse]:
    stmt = select(CitizenReport).order_by(CitizenReport.created_at.desc()).limit(limit)
    if ward_id is not None:
        stmt = (
            select(CitizenReport)
            .where(CitizenReport.ward_id == ward_id)
            .order_by(CitizenReport.created_at.desc())
            .limit(limit)
        )
    rows = db.scalars(stmt).all()
    return [
        CitizenReportResponse(
            id=item.id,
            ward_id=item.ward_id,
            text=item.text,
            language=item.language,
            category=item.category,
            sentiment_score=item.sentiment_score,
            priority_weight=item.priority_weight,
            created_at=item.created_at,
        )
        for item in rows
    ]
