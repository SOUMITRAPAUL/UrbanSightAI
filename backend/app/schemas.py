from __future__ import annotations

from datetime import datetime
from typing import Literal, Any

from pydantic import BaseModel, Field


Role = Literal["planner", "enumerator", "viewer"]
Severity = Literal["low", "medium", "high", "critical"]
PlanningStrategy = Literal[
    "balanced", "climate_resilience", "equity_first", "fast_delivery", "custom"
]


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=64)
    role: Role = "viewer"


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: Role


class UserProfile(BaseModel):
    username: str
    role: Role


class WardSummary(BaseModel):
    id: int
    code: str
    name: str
    area_km2: float
    population: int
    households: int
    last_updated: datetime
    sdg11_score: float


class WardIndicatorResponse(BaseModel):
    informal_area_pct: float
    blocked_drain_count: int
    green_deficit_index: float
    flood_risk_index: float
    sdg11_score: float
    exposed_population: int


class WardDigitalTwin(BaseModel):
    ward: WardSummary
    indicators: WardIndicatorResponse
    boundary: list[list[float]]
    feature_counts: dict[str, int]
    integrity_notes: list[str] = []


class TwinHouse(BaseModel):
    lat: float
    lon: float
    height: float
    footprint: float


class TwinAreaAsset(BaseModel):
    lat: float
    lon: float
    size: float


class TwinBlockedDrain(BaseModel):
    lat: float
    lon: float
    severity: Severity
    risk_score: float
    label: str


class TwinBlockedDrainSegment(BaseModel):
    path: list[list[float]]
    severity: Severity
    risk_score: float
    label: str


class TwinRiskZone(BaseModel):
    lat: float
    lon: float
    radius: float
    risk_score: float
    label: str


class TwinInformalZone(BaseModel):
    polygon: list[list[float]]
    density_score: float
    households_est: int


class TwinLayers(BaseModel):
    roads: list[list[list[float]]]
    drains: list[list[list[float]]]
    rivers: list[list[list[float]]]
    blocked_drain_network: list[TwinBlockedDrainSegment]
    waterbodies: list[list[list[float]]]
    houses: list[TwinHouse]
    playgrounds: list[TwinAreaAsset]
    parks: list[TwinAreaAsset]
    blocked_drains: list[TwinBlockedDrain]
    flood_zones: list[TwinRiskZone]
    informal_zones: list[TwinInformalZone]
    summary: dict[str, int]


class TwinProblem(BaseModel):
    issue: str
    severity: Severity
    value: float
    summary: str


class TwinHotspot(BaseModel):
    lat: float
    lon: float
    issue: str
    severity: Severity
    source: str


class TwinAction(BaseModel):
    intervention_id: int
    title: str
    category: str
    agency: str
    status: Literal["planned", "in_progress", "completed"]
    progress_pct: int
    estimated_cost_lakh: float
    expected_beneficiaries: int


class PublicDataSources(BaseModel):
    boundaries_source: str
    map_features_source: str
    weather_source: str
    weather_status: str
    weather_observed_at: datetime | None
    rainfall_mm: float | None
    temperature_c: float | None
    river_segments: int
    generated_at: datetime


class LiveWeatherSnapshot(BaseModel):
    source: str
    status: str
    observed_at: datetime | None
    rainfall_mm: float | None
    temperature_c: float | None
    wind_speed_kmh: float | None


class WardTwinSceneResponse(BaseModel):
    ward: WardSummary
    boundary: list[list[float]]
    layers: TwinLayers
    feature_counts: dict[str, int]
    scores: WardIndicatorResponse
    data_sources: PublicDataSources
    problems: list[TwinProblem]
    actions_taken: list[TwinAction]
    hotspots: list[TwinHotspot]
    integrity_notes: list[str] = []


class SDG11TargetStatus(BaseModel):
    target: str
    score: float
    status: Literal["on_track", "watch", "critical"]
    evidence: str


class SDG11GovernanceCard(BaseModel):
    ward_id: int
    ward_name: str
    overall_score: float
    targets: list[SDG11TargetStatus]
    priority_message: str
    generated_at: datetime


class InterAgencyTask(BaseModel):
    intervention_id: int
    action_title: str
    category: str
    lead_agency: str
    supporting_agencies: list[str]
    permit_required: bool
    estimated_cost_lakh: float
    timeline_weeks: int
    dependency: str


class InterAgencyPacket(BaseModel):
    ward_id: int
    ward_name: str
    generated_at: datetime
    checklist: list[str]
    tasks: list[InterAgencyTask]


class AuditTrailItem(BaseModel):
    id: int
    timestamp: datetime
    actor_username: str
    actor_role: str
    action: str
    ward_id: int | None
    details: dict[str, object]


class AIComponentStatus(BaseModel):
    name: str
    technology: str
    policy_output: str
    status: Literal["implemented", "pilot", "planned"]
    endpoint: str


class WorkflowSourceRecord(BaseModel):
    name: str
    kind: str
    status: Literal["implemented", "partial", "missing"]
    records: int
    provenance: str
    last_updated: datetime | None
    notes: str


class WorkflowStageStatus(BaseModel):
    id: str
    title: str
    status: Literal["implemented", "partial", "missing"]
    summary: str
    metrics: dict[str, float | int | str]
    details: list[str]


class WorkflowCandidate(BaseModel):
    title: str
    category: str
    agency: str
    permit_required: bool
    rough_cost_lakh: float
    expected_beneficiaries: int
    trigger_metric: str
    rationale: str
    evidence_sources: list[str]
    mapped_intervention_id: int | None = None
    mapped_intervention_title: str | None = None
    mapped_ranking_score: float | None = None


class WardWorkflowResponse(BaseModel):
    ward: WardSummary
    generated_at: datetime
    workflow_complete: bool
    stages: list[WorkflowStageStatus]
    source_inventory: list[WorkflowSourceRecord]
    candidate_micro_works: list[WorkflowCandidate]


class TopWorkItem(BaseModel):
    id: int
    title: str
    category: str
    agency: str
    permit_required: bool
    estimated_cost_lakh: float
    expected_beneficiaries: int
    beneficiary_ci_low: int
    beneficiary_ci_high: int
    beneficiary_method: str
    impact_per_lakh: float
    ranking_score: float
    feasibility: float
    equity_need: float
    urgency: float
    justification: str


class TopWorklistResponse(BaseModel):
    ward_id: int
    generated_at: datetime
    items: list[TopWorkItem]


class ScenarioRequest(BaseModel):
    ward_id: int
    budget_lakh: float = Field(gt=0)


class ScenarioAgencyLoad(BaseModel):
    agency: str
    projects: int
    budget_lakh: float
    opex_lakh: float
    avg_timeline_months: float
    permit_projects: int
    share_pct: float


class ScenarioRoadmapItem(BaseModel):
    intervention_id: int
    title: str
    category: str
    agency: str
    phase: str
    start_month: int
    end_month: int
    dependency: str
    delivery_status: Literal["ready", "permit_gating", "dependency_gating", "capacity_queued"]


class ScenarioTradeoffAlert(BaseModel):
    severity: Literal["info", "warning", "critical"]
    topic: str
    message: str


class ScenarioDeferredProject(BaseModel):
    intervention_id: int
    title: str
    category: str
    agency: str
    estimated_cost_lakh: float
    utility_density: float
    primary_constraint: str
    action_hint: str


class StrategyComparisonItem(BaseModel):
    strategy: PlanningStrategy
    label: str
    selected_projects: int
    used_budget_lakh: float
    impacted_households: int
    estimated_sdg11_gain: float
    avg_timeline_months: float
    permit_share_pct: float
    readiness_score: float


class ScenarioResult(BaseModel):
    ward_id: int
    budget_lakh: float
    strategy_profile: PlanningStrategy
    strategy_label: str
    strategy_description: str
    selected_projects: list[TopWorkItem]
    used_budget_lakh: float
    remaining_budget_lakh: float
    impacted_households: int
    estimated_sdg11_gain: float
    selection_method: str
    decision_basis: dict[str, float | int | str]
    portfolio_summary: dict[str, float | int | str]
    selected_reasoning: list[dict[str, object]]
    agency_load: list[ScenarioAgencyLoad]
    implementation_roadmap: list[ScenarioRoadmapItem]
    tradeoff_alerts: list[ScenarioTradeoffAlert]
    deferred_projects: list[ScenarioDeferredProject]
    counterfactuals: list[dict[str, float]]
    strategy_comparison: list[StrategyComparisonItem]


class CitizenReportRequest(BaseModel):
    ward_id: int
    text: str = Field(min_length=5, max_length=400)
    language: Literal["bangla", "english", "mixed"] = "mixed"


class CitizenReportPrediction(BaseModel):
    category: str
    confidence: float
    sentiment_score: float
    priority_weight: float


class CitizenReportResponse(BaseModel):
    id: int
    ward_id: int
    text: str
    language: str
    category: str
    sentiment_score: float
    priority_weight: float
    created_at: datetime


class ODKSubmissionRequest(BaseModel):
    ward_id: int
    text: str = Field(min_length=5, max_length=400)
    language: Literal["bangla", "english", "mixed"] = "mixed"
    issue_type: Literal["blocked_drain", "flooding", "waste", "road_damage", "water_supply", "auto"] = "auto"
    location_hint: str = Field(default="", max_length=120)
    reporter_name: str = Field(default="", max_length=60)
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class NotificationFeedItem(BaseModel):
    id: int
    timestamp: datetime
    source: str
    severity: Literal["info", "success", "warning", "critical"]
    message: str
    ward_id: int | None


class LiveSourceEvent(BaseModel):
    source: Literal["citizen_portal", "odk_kobo", "sensor_feed", "open_meteo", "model_rerun"]
    records_ingested: int
    note: str


class LiveIngestRequest(BaseModel):
    citizen_reports: int = Field(default=3, ge=0, le=30)
    odk_forms: int = Field(default=2, ge=0, le=20)
    sensor_pulses: int = Field(default=1, ge=0, le=20)


class LiveUpdateResponse(BaseModel):
    ward_id: int
    triggered_at: datetime
    sources: list[LiveSourceEvent]
    updated_indicators: WardIndicatorResponse
    live_weather: LiveWeatherSnapshot | None = None
    top_changes: list[str]
class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    ward_id: int | None = None
    chat_mode: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

class ConsultRequest(BaseModel):
    vision: str | None = Field(default=None, max_length=500)
    budget_lakh: float = Field(default=8.0, gt=0.4, le=100.0)
    sector_priorities: dict[str, float] | None = None

class ConsultResponse(BaseModel):
    vision: str
    weights: dict[str, float]
    reasoning: str
    result: ScenarioResult
    budget_plan: dict[str, Any]
