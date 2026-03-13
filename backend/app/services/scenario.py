from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import ceil
from typing import Any

import numpy as np


CATEGORY_BASE_TIMELINE = {
    "Drainage": 3.0,
    "Water": 5.0,
    "Waste": 2.0,
    "Road": 4.0,
    "Green": 4.0,
    "Public Safety": 2.5,
}

CATEGORY_OPEX_RATIO = {
    "Drainage": 0.08,
    "Water": 0.10,
    "Waste": 0.12,
    "Road": 0.06,
    "Green": 0.07,
    "Public Safety": 0.09,
}

CLIMATE_PRIORITY = {"Drainage", "Water", "Waste", "Green"}


@dataclass(frozen=True)
class StrategyProfile:
    key: str
    label: str
    description: str
    weights: dict[str, float]
    category_bonus: dict[str, float]
    climate_bonus: float
    reserve_ratio: float
    opex_cap_ratio: float
    timeline_factor: float
    agency_capacity_bonus: int
    permit_share_cap: float
    seed_categories: tuple[str, ...]
    category_soft_cap: float = 0.58


STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "balanced": StrategyProfile(
        key="balanced",
        label="Balanced Portfolio",
        description="Blends impact, equity, urgency, and delivery readiness for a steady ward program.",
        weights={
            "impact_per_lakh": 0.26,
            "equity_need": 0.18,
            "urgency": 0.16,
            "feasibility": 0.12,
            "beneficiary_norm": 0.10,
            "prior_rank_norm": 0.08,
            "readiness_norm": 0.10,
        },
        category_bonus={"Drainage": 0.02, "Waste": 0.015},
        climate_bonus=0.04,
        reserve_ratio=0.10,
        opex_cap_ratio=0.22,
        timeline_factor=1.0,
        agency_capacity_bonus=0,
        permit_share_cap=0.42,
        seed_categories=("Drainage", "Water", "Waste"),
        category_soft_cap=0.58,
    ),
    "climate_resilience": StrategyProfile(
        key="climate_resilience",
        label="Climate Resilience",
        description="Prioritizes flood, drainage, canal, and green interventions that reduce climate exposure.",
        weights={
            "impact_per_lakh": 0.22,
            "equity_need": 0.14,
            "urgency": 0.20,
            "feasibility": 0.10,
            "beneficiary_norm": 0.08,
            "prior_rank_norm": 0.08,
            "readiness_norm": 0.18,
        },
        category_bonus={"Drainage": 0.06, "Water": 0.06, "Green": 0.05, "Waste": 0.03},
        climate_bonus=0.10,
        reserve_ratio=0.12,
        opex_cap_ratio=0.25,
        timeline_factor=1.12,
        agency_capacity_bonus=0,
        permit_share_cap=0.48,
        seed_categories=("Drainage", "Water", "Green"),
        category_soft_cap=0.65,
    ),
    "equity_first": StrategyProfile(
        key="equity_first",
        label="Equity First",
        description="Favors high-need, high-beneficiary interventions even when delivery friction is slightly higher.",
        weights={
            "impact_per_lakh": 0.18,
            "equity_need": 0.30,
            "urgency": 0.14,
            "feasibility": 0.08,
            "beneficiary_norm": 0.16,
            "prior_rank_norm": 0.06,
            "readiness_norm": 0.08,
        },
        category_bonus={"Drainage": 0.03, "Waste": 0.05, "Green": 0.05, "Public Safety": 0.04},
        climate_bonus=0.05,
        reserve_ratio=0.09,
        opex_cap_ratio=0.24,
        timeline_factor=1.2,
        agency_capacity_bonus=0,
        permit_share_cap=0.52,
        seed_categories=("Drainage", "Waste", "Green"),
        category_soft_cap=0.62,
    ),
    "fast_delivery": StrategyProfile(
        key="fast_delivery",
        label="Fast Delivery",
        description="Biases toward permit-light, high-readiness projects that can move inside one municipal cycle.",
        weights={
            "impact_per_lakh": 0.20,
            "equity_need": 0.12,
            "urgency": 0.18,
            "feasibility": 0.18,
            "beneficiary_norm": 0.08,
            "prior_rank_norm": 0.06,
            "readiness_norm": 0.18,
        },
        category_bonus={"Drainage": 0.04, "Waste": 0.05, "Public Safety": 0.06},
        climate_bonus=0.02,
        reserve_ratio=0.08,
        opex_cap_ratio=0.18,
        timeline_factor=0.72,
        agency_capacity_bonus=1,
        permit_share_cap=0.28,
        seed_categories=("Drainage", "Waste", "Public Safety"),
        category_soft_cap=0.52,
    ),
}

UTILITY_WEIGHTS = STRATEGY_PROFILES["balanced"].weights


@dataclass
class ScenarioConstraints:
    budget_lakh: float
    reserve_lakh: float
    investable_budget_lakh: float
    opex_cap_lakh: float
    target_timeline_months: float
    agency_capacity_per_agency: int
    permit_share_cap: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _strategy_profile(strategy: str | None, custom_weights: dict[str, float] | None = None) -> StrategyProfile:
    key = str(strategy or "balanced").strip().lower()
    profile = STRATEGY_PROFILES.get(key, STRATEGY_PROFILES["balanced"])
    if custom_weights:
        return replace(
            profile,
            key="custom",
            label="Custom AI Vision",
            description="Dynamically generated profile based on user consultation.",
            weights={**profile.weights, **custom_weights},
        )
    return profile


def _priority_score(item: dict[str, Any]) -> float:
    if "utility_density" in item:
        return float(item["utility_density"])
    score = float(item.get("ranking_score", 0.0))
    cost = max(float(item.get("estimated_cost_lakh", 0.0)), 0.12)
    return score / cost


def _budget_constraints(
    budget_lakh: float,
    profile: StrategyProfile,
) -> ScenarioConstraints:
    reserve_lakh = float(max(0.35, budget_lakh * profile.reserve_ratio))
    investable_budget = float(max(0.1, budget_lakh - reserve_lakh))
    opex_cap = float(max(0.3, budget_lakh * profile.opex_cap_ratio))
    base_timeline = 12.0 if budget_lakh <= 25 else 16.0
    target_timeline = float(base_timeline * profile.timeline_factor)
    agency_capacity = max(2, int(round(budget_lakh / 6.0)) + profile.agency_capacity_bonus)
    return ScenarioConstraints(
        budget_lakh=float(budget_lakh),
        reserve_lakh=reserve_lakh,
        investable_budget_lakh=investable_budget,
        opex_cap_lakh=opex_cap,
        target_timeline_months=target_timeline,
        agency_capacity_per_agency=agency_capacity,
        permit_share_cap=profile.permit_share_cap,
    )


def _enrich_items(items: list[dict[str, Any]], profile: StrategyProfile) -> list[dict[str, Any]]:
    if not items:
        return []

    max_impact = max(_safe_float(it.get("impact_per_lakh"), 0.0) for it in items)
    max_rank = max(_safe_float(it.get("ranking_score"), 0.0) for it in items)
    max_beneficiaries = max(_safe_int(it.get("expected_beneficiaries"), 0) for it in items)

    enriched: list[dict[str, Any]] = []
    for item in items:
        category = str(item.get("category", "Other"))
        agency = str(item.get("agency", "Unknown"))
        cost = max(_safe_float(item.get("estimated_cost_lakh"), 0.0), 0.12)
        impact_per_lakh = max(_safe_float(item.get("impact_per_lakh"), 0.0), 0.0)
        ranking = max(_safe_float(item.get("ranking_score"), 0.0), 0.0)
        beneficiaries = max(_safe_int(item.get("expected_beneficiaries"), 0), 0)
        feasibility = float(np.clip(_safe_float(item.get("feasibility"), 0.5), 0.0, 1.0))
        equity = float(np.clip(_safe_float(item.get("equity_need"), 0.5), 0.0, 1.0))
        urgency = float(np.clip(_safe_float(item.get("urgency"), 0.5), 0.0, 1.0))
        permit_required = bool(item.get("permit_required", False))

        base_timeline = CATEGORY_BASE_TIMELINE.get(category, 4.0)
        execution_months = (
            base_timeline
            + cost * 0.52
            + (1.6 if permit_required else 0.0)
            + max(0.0, 1.0 - feasibility) * 1.7
        )
        opex_lakh = cost * CATEGORY_OPEX_RATIO.get(category, 0.08)

        impact_norm = impact_per_lakh / max(max_impact, 1.0)
        rank_norm = ranking / max(max_rank, 1.0)
        beneficiary_norm = beneficiaries / max(max_beneficiaries, 1)
        readiness_norm = float(
            np.clip(
                feasibility * 0.55
                + (0.25 if not permit_required else 0.0)
                + np.clip(1.0 - execution_months / 12.0, 0.0, 1.0) * 0.20,
                0.0,
                1.0,
            )
        )
        strategic_bonus = float(profile.category_bonus.get(category, 0.0))
        if category in CLIMATE_PRIORITY:
            strategic_bonus += profile.climate_bonus

        utility_score = (
            profile.weights["impact_per_lakh"] * impact_norm
            + profile.weights["equity_need"] * equity
            + profile.weights["urgency"] * urgency
            + profile.weights["feasibility"] * feasibility
            + profile.weights["beneficiary_norm"] * beneficiary_norm
            + profile.weights["prior_rank_norm"] * rank_norm
            + profile.weights["readiness_norm"] * readiness_norm
            + strategic_bonus
        )
        utility_density = utility_score / (cost + 0.08)
        enriched.append(
            {
                **item,
                "category": category,
                "agency": agency,
                "estimated_cost_lakh": float(cost),
                "expected_beneficiaries": int(beneficiaries),
                "feasibility": feasibility,
                "equity_need": equity,
                "urgency": urgency,
                "permit_required": permit_required,
                "execution_months": float(round(float(execution_months), 2)),
                "opex_lakh": float(round(float(opex_lakh), 3)),
                "utility_score": float(round(float(utility_score), 6)),
                "utility_density": float(round(float(utility_density), 6)),
                "strategic_bonus": float(round(float(strategic_bonus), 4)),
                "impact_norm": float(round(float(impact_norm), 6)),
                "rank_norm": float(round(float(rank_norm), 6)),
                "beneficiary_norm": float(round(float(beneficiary_norm), 6)),
                "readiness_norm": float(round(float(readiness_norm), 6)),
            }
        )
    return enriched


def _selection_failure_reason(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    constraints: ScenarioConstraints,
    agency_counter: Counter[str],
    category_counter: Counter[str],
    profile: StrategyProfile,
) -> str | None:
    current_cost = sum(_safe_float(it.get("estimated_cost_lakh")) for it in selected)
    current_opex = sum(_safe_float(it.get("opex_lakh")) for it in selected)
    current_permit = sum(1 for it in selected if bool(it.get("permit_required")))
    current_timeline_sum = sum(_safe_float(it.get("execution_months")) for it in selected)

    candidate_cost = _safe_float(candidate.get("estimated_cost_lakh"))
    candidate_opex = _safe_float(candidate.get("opex_lakh"))
    candidate_agency = str(candidate.get("agency", "Unknown"))
    candidate_category = str(candidate.get("category", "Other"))
    candidate_permit = bool(candidate.get("permit_required", False))
    candidate_urgency = _safe_float(candidate.get("urgency"), 0.5)
    candidate_density = _safe_float(candidate.get("utility_density"))
    candidate_readiness = _safe_float(candidate.get("readiness_norm"))

    if current_cost + candidate_cost > constraints.investable_budget_lakh + 1e-9:
        return "budget"
    if current_opex + candidate_opex > constraints.opex_cap_lakh + 1e-9:
        return "opex"
    if agency_counter[candidate_agency] >= constraints.agency_capacity_per_agency:
        return "agency_capacity"

    next_count = len(selected) + 1
    next_permit_share = (current_permit + (1 if candidate_permit else 0)) / max(next_count, 1)
    if (
        candidate_permit
        and next_permit_share > constraints.permit_share_cap
        and candidate_urgency < 0.84
    ):
        return "permit_share"

    if len(selected) >= 2:
        current_share = category_counter[candidate_category] / max(len(selected), 1)
        if current_share > profile.category_soft_cap and candidate_density < 0.80:
            return "category_concentration"

    next_timeline_avg = (
        current_timeline_sum + _safe_float(candidate.get("execution_months"))
    ) / max(next_count, 1)
    if next_timeline_avg > constraints.target_timeline_months * 1.15 and candidate_urgency < 0.80:
        return "timeline"

    if profile.key == "fast_delivery" and candidate_readiness < 0.56 and candidate_urgency < 0.82:
        return "readiness"

    return None


def _realworld_select(
    items: list[dict[str, Any]],
    budget_lakh: float,
    profile: StrategyProfile,
) -> tuple[list[dict[str, Any]], ScenarioConstraints, dict[str, int], dict[int, str]]:
    constraints = _budget_constraints(budget_lakh, profile)
    if not items:
        return [], constraints, {}, {}

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    agency_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    skipped: defaultdict[str, int] = defaultdict(int)
    fail_reasons: dict[int, str] = {}

    seed_candidates = sorted(
        [
            it
            for it in items
            if str(it.get("category")) in set(profile.seed_categories)
            and _safe_float(it.get("urgency")) >= 0.60
        ],
        key=_priority_score,
        reverse=True,
    )
    seeded_cat: set[str] = set()
    for candidate in seed_candidates:
        category = str(candidate.get("category"))
        candidate_id = _safe_int(candidate.get("id"))
        if category in seeded_cat or candidate_id in selected_ids:
            continue
        failure = _selection_failure_reason(
            candidate,
            selected=selected,
            constraints=constraints,
            agency_counter=agency_counter,
            category_counter=category_counter,
            profile=profile,
        )
        if failure is None:
            selected.append(candidate)
            selected_ids.add(candidate_id)
            agency_counter[str(candidate.get("agency", "Unknown"))] += 1
            category_counter[category] += 1
            seeded_cat.add(category)
        else:
            skipped[failure] += 1
            fail_reasons.setdefault(candidate_id, failure)
        if len(seeded_cat) >= 2:
            break

    for candidate in sorted(items, key=_priority_score, reverse=True):
        candidate_id = _safe_int(candidate.get("id"))
        if candidate_id in selected_ids:
            continue
        failure = _selection_failure_reason(
            candidate,
            selected=selected,
            constraints=constraints,
            agency_counter=agency_counter,
            category_counter=category_counter,
            profile=profile,
        )
        if failure is None:
            selected.append(candidate)
            selected_ids.add(candidate_id)
            agency_counter[str(candidate.get("agency", "Unknown"))] += 1
            category_counter[str(candidate.get("category", "Other"))] += 1
        else:
            skipped[failure] += 1
            fail_reasons.setdefault(candidate_id, failure)

    if len(selected) <= 3:
        relaxed_permit_cap = min(0.60, constraints.permit_share_cap + 0.10)
        for candidate in sorted(items, key=_priority_score, reverse=True):
            candidate_id = _safe_int(candidate.get("id"))
            if candidate_id in selected_ids:
                continue
            current_cost = sum(_safe_float(it.get("estimated_cost_lakh")) for it in selected)
            current_opex = sum(_safe_float(it.get("opex_lakh")) for it in selected)
            current_permit = sum(1 for it in selected if bool(it.get("permit_required")))
            next_count = len(selected) + 1
            next_permit_share = (
                current_permit + (1 if bool(candidate.get("permit_required")) else 0)
            ) / max(next_count, 1)
            if current_cost + _safe_float(candidate.get("estimated_cost_lakh")) > constraints.investable_budget_lakh:
                continue
            if current_opex + _safe_float(candidate.get("opex_lakh")) > constraints.opex_cap_lakh:
                continue
            if next_permit_share > relaxed_permit_cap and _safe_float(candidate.get("urgency")) < 0.9:
                continue
            candidate_agency = str(candidate.get("agency", "Unknown"))
            if agency_counter[candidate_agency] >= constraints.agency_capacity_per_agency + 1:
                continue
            selected.append(candidate)
            agency_counter[candidate_agency] += 1
            category_counter[str(candidate.get("category", "Other"))] += 1
            selected_ids.add(candidate_id)

    return selected, constraints, dict(skipped), fail_reasons


def _selected_reasoning(
    selected: list[dict[str, Any]],
    ranked_items: list[dict[str, Any]],
    profile: StrategyProfile,
) -> list[dict[str, object]]:
    if not selected:
        return []
    by_id_rank = {int(item["id"]): idx + 1 for idx, item in enumerate(ranked_items)}
    rows: list[dict[str, object]] = []
    for item in selected:
        reasons: list[str] = []
        if _safe_float(item.get("impact_norm")) >= 0.65:
            reasons.append("High impact-per-budget efficiency")
        if _safe_float(item.get("equity_need")) >= 0.70:
            reasons.append("Strong equity targeting")
        if _safe_float(item.get("urgency")) >= 0.70:
            reasons.append("Urgent risk-reduction need")
        if _safe_float(item.get("readiness_norm")) >= 0.68:
            reasons.append("High delivery readiness")
        if not bool(item.get("permit_required")):
            reasons.append("Low implementation friction (no permit)")
        if _safe_float(item.get("execution_months")) <= 8.0:
            reasons.append("Fits short municipal delivery timeline")
        if str(item.get("category")) in CLIMATE_PRIORITY and profile.key == "climate_resilience":
            reasons.append("Matches climate-resilience strategy")
        if profile.key == "equity_first" and _safe_float(item.get("equity_need")) >= 0.65:
            reasons.append("Advances equity-first targeting")
        if profile.key == "fast_delivery" and _safe_float(item.get("readiness_norm")) >= 0.6:
            reasons.append("Ready for fast-delivery sequencing")
        if not reasons:
            reasons.append("Balanced multi-criteria utility under constraints")
        rows.append(
            {
                "intervention_id": int(item["id"]),
                "title": str(item["title"]),
                "category": str(item["category"]),
                "estimated_cost_lakh": float(item["estimated_cost_lakh"]),
                "expected_beneficiaries": int(item["expected_beneficiaries"]),
                "beneficiary_ci_low": int(item.get("beneficiary_ci_low", item["expected_beneficiaries"])),
                "beneficiary_ci_high": int(item.get("beneficiary_ci_high", item["expected_beneficiaries"])),
                "impact_per_lakh": float(item.get("impact_per_lakh", 0.0)),
                "utility_score": float(item.get("utility_score", 0.0)),
                "utility_density": float(item.get("utility_density", 0.0)),
                "execution_months": float(item.get("execution_months", 0.0)),
                "opex_lakh": float(item.get("opex_lakh", 0.0)),
                "readiness_norm": float(item.get("readiness_norm", 0.0)),
                "cost_efficiency_rank": int(by_id_rank.get(int(item["id"]), 999)),
                "reasons": reasons,
            }
        )
    return rows


def _build_agency_load(selected: list[dict[str, Any]]) -> list[dict[str, object]]:
    if not selected:
        return []
    total_budget = max(sum(_safe_float(item.get("estimated_cost_lakh")) for item in selected), 0.1)
    by_agency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_agency[str(item.get("agency", "Unknown"))].append(item)

    rows: list[dict[str, object]] = []
    for agency, items in by_agency.items():
        budget = sum(_safe_float(item.get("estimated_cost_lakh")) for item in items)
        opex = sum(_safe_float(item.get("opex_lakh")) for item in items)
        avg_timeline = float(np.mean([_safe_float(item.get("execution_months")) for item in items]))
        permit_projects = sum(1 for item in items if bool(item.get("permit_required")))
        rows.append(
            {
                "agency": agency,
                "projects": len(items),
                "budget_lakh": round(float(budget), 3),
                "opex_lakh": round(float(opex), 3),
                "avg_timeline_months": round(avg_timeline, 2),
                "permit_projects": permit_projects,
                "share_pct": round(float(budget / total_budget * 100.0), 2),
            }
        )
    rows.sort(key=lambda item: (float(item["budget_lakh"]), int(item["projects"])), reverse=True)
    return rows


def _phase_label(month: int) -> str:
    if month <= 3:
        return "0-3 months"
    if month <= 6:
        return "4-6 months"
    if month <= 9:
        return "7-9 months"
    return "10-12 months"


def _build_roadmap(selected: list[dict[str, Any]]) -> list[dict[str, object]]:
    if not selected:
        return []

    ordered = sorted(
        selected,
        key=lambda item: (
            bool(item.get("permit_required", False)),
            -_safe_float(item.get("urgency")),
            -_safe_float(item.get("utility_density")),
        ),
    )
    agency_next_month: defaultdict[str, int] = defaultdict(lambda: 1)
    category_last_end: defaultdict[str, int] = defaultdict(int)
    roadmap: list[dict[str, object]] = []

    for item in ordered:
        category = str(item.get("category", "Other"))
        agency = str(item.get("agency", "Unknown"))
        exec_months = max(1, int(ceil(_safe_float(item.get("execution_months"), 4.0))))
        urgency = _safe_float(item.get("urgency"), 0.5)
        permit_required = bool(item.get("permit_required", False))

        dependency = "Ready for immediate mobilization"
        dependency_month = 1
        if permit_required:
            dependency = "Permit screening and procurement approval required"
            dependency_month = 3
        elif category == "Road" and category_last_end["Drainage"] > 0:
            dependency = "Sequence after drainage works on the target corridor"
            dependency_month = category_last_end["Drainage"] + 1
        elif category == "Green" and max(category_last_end["Drainage"], category_last_end["Water"]) > 0:
            dependency = "Sequence after drainage/water stabilization and site clearance"
            dependency_month = max(category_last_end["Drainage"], category_last_end["Water"]) + 1
        elif category == "Public Safety" and max(category_last_end["Road"], category_last_end["Drainage"]) > 0:
            dependency = "Coordinate after civil access routes are confirmed"
            dependency_month = max(category_last_end["Road"], category_last_end["Drainage"]) + 1

        if permit_required:
            base_start = 4
        elif exec_months >= 9:
            base_start = 4
        elif urgency >= 0.72:
            base_start = 1
        elif exec_months >= 6:
            base_start = 3
        else:
            base_start = 2

        start_month = max(base_start, dependency_month, agency_next_month[agency])
        end_month = min(18, start_month + exec_months - 1)
        if permit_required:
            delivery_status = "permit_gating"
        elif dependency != "Ready for immediate mobilization":
            delivery_status = "dependency_gating"
        elif start_month >= 7:
            delivery_status = "capacity_queued"
        else:
            delivery_status = "ready"

        roadmap.append(
            {
                "intervention_id": int(item["id"]),
                "title": str(item["title"]),
                "category": category,
                "agency": agency,
                "phase": _phase_label(start_month),
                "start_month": int(start_month),
                "end_month": int(end_month),
                "dependency": dependency,
                "delivery_status": delivery_status,
            }
        )
        agency_next_month[agency] = end_month + 1
        category_last_end[category] = max(category_last_end[category], end_month)

    roadmap.sort(key=lambda row: (int(row["start_month"]), int(row["end_month"]), str(row["agency"])))
    return roadmap


def _build_portfolio_summary(
    selected: list[dict[str, Any]],
    constraints: ScenarioConstraints,
    roadmap: list[dict[str, object]],
) -> dict[str, float | int | str]:
    if not selected:
        return {
            "selected_count": 0,
            "agencies_engaged": 0,
            "budget_utilization_pct": 0.0,
            "opex_utilization_pct": 0.0,
            "permit_share_pct": 0.0,
            "climate_share_pct": 0.0,
            "avg_timeline_months": 0.0,
            "critical_path_months": 0,
            "households_per_lakh": 0.0,
            "readiness_score": 0.0,
            "dominant_category": "none",
            "delivery_risk": "high",
        }

    used_budget = sum(_safe_float(item.get("estimated_cost_lakh")) for item in selected)
    used_opex = sum(_safe_float(item.get("opex_lakh")) for item in selected)
    permit_share = sum(1 for item in selected if bool(item.get("permit_required"))) / max(len(selected), 1)
    climate_share = sum(1 for item in selected if str(item.get("category")) in CLIMATE_PRIORITY) / max(len(selected), 1)
    avg_timeline = float(np.mean([_safe_float(item.get("execution_months")) for item in selected]))
    readiness_score = float(np.mean([_safe_float(item.get("readiness_norm")) for item in selected]))
    households = int(sum(_safe_int(item.get("expected_beneficiaries")) for item in selected) / 4.6)
    dominant_category = Counter(str(item.get("category", "Other")) for item in selected).most_common(1)[0][0]
    critical_path = max((int(step["end_month"]) for step in roadmap), default=0)

    if permit_share > 0.45 or avg_timeline > constraints.target_timeline_months * 1.05:
        delivery_risk = "high"
    elif permit_share > 0.28 or readiness_score < 0.58:
        delivery_risk = "medium"
    else:
        delivery_risk = "low"

    return {
        "selected_count": len(selected),
        "agencies_engaged": len({str(item.get("agency", "Unknown")) for item in selected}),
        "budget_utilization_pct": round(float(used_budget / max(constraints.investable_budget_lakh, 0.1) * 100.0), 2),
        "opex_utilization_pct": round(float(used_opex / max(constraints.opex_cap_lakh, 0.1) * 100.0), 2),
        "permit_share_pct": round(float(permit_share * 100.0), 2),
        "climate_share_pct": round(float(climate_share * 100.0), 2),
        "avg_timeline_months": round(avg_timeline, 2),
        "critical_path_months": int(critical_path),
        "households_per_lakh": round(float(households / max(used_budget, 0.1)), 2),
        "readiness_score": round(readiness_score, 3),
        "dominant_category": dominant_category,
        "delivery_risk": delivery_risk,
    }


def _build_tradeoff_alerts(
    selected: list[dict[str, Any]],
    constraints: ScenarioConstraints,
    skipped: dict[str, int],
    agency_load: list[dict[str, object]],
    portfolio_summary: dict[str, float | int | str],
    profile: StrategyProfile,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if not selected:
        return [
            {
                "severity": "critical",
                "topic": "Selection Failure",
                "message": "No projects fit the current budget and implementation constraints.",
            }
        ]

    permit_share = _safe_float(portfolio_summary.get("permit_share_pct")) / 100.0
    readiness_score = _safe_float(portfolio_summary.get("readiness_score"))
    dominant_category = str(portfolio_summary.get("dominant_category", "none"))
    budget_utilization = _safe_float(portfolio_summary.get("budget_utilization_pct"))

    if permit_share > constraints.permit_share_cap * 0.92:
        alerts.append(
            {
                "severity": "warning",
                "topic": "Permit Exposure",
                "message": (
                    f"Selected portfolio consumes {permit_share*100:.1f}% of the permit cap. "
                    "Implementation may stall if regulatory approvals are delayed."
                ),
            }
        )
    if agency_load and _safe_float(agency_load[0].get("share_pct")) > 45.0:
        alerts.append(
            {
                "severity": "warning",
                "topic": "Agency Concentration",
                "message": (
                    f"{agency_load[0]['agency']} carries {agency_load[0]['share_pct']}% of the total spend. "
                    "Large workload on a single agency increases delivery risk."
                ),
            }
        )
    if budget_utilization < 72.0:
        alerts.append(
            {
                "severity": "info",
                "topic": "Budget Slack",
                "message": (
                    f"Only {budget_utilization:.1f}% of the budget is used. "
                    "Consider relaxing readiness or agency constraints to admit more projects."
                ),
            }
        )
    if readiness_score < 0.56:
        alerts.append(
            {
                "severity": "warning",
                "topic": "Delivery Readiness",
                "message": (
                    "High number of complex or permit-heavy projects. "
                    "Immediate project preparation and field design are required to avoid delays."
                ),
            }
        )
    if profile.key == "climate_resilience" and _safe_float(portfolio_summary.get("climate_share_pct")) < 55.0:
        alerts.append(
            {
                "severity": "warning",
                "topic": "Climate Alignment",
                "message": (
                    "Resilience focus is lower than expected. "
                    "Consider increasing budget to include larger flood-mitigation works."
                ),
            }
        )
    if profile.key == "equity_first":
        avg_equity = float(np.mean([_safe_float(item.get("equity_need")) for item in selected]))
        if avg_equity < 0.64:
            alerts.append(
                {
                    "severity": "warning",
                    "topic": "Equity Targeting",
                    "message": "Equity-first strategy is constrained by project readiness; high-need zones may be underserved.",
                }
            )
    if skipped.get("budget", 0) > 0:
        alerts.append(
            {
                "severity": "info",
                "topic": "Deferred Demand",
                "message": f"Budget limit reached; {int(skipped['budget'])} viable projects were deferred to the next cycle.",
            }
        )
    if dominant_category != "none":
        alerts.append(
            {
                "severity": "info",
                "topic": "Portfolio Shape",
                "message": f"Investment is primarily concentrated in {dominant_category.lower()} improvements.",
            }
        )
    return alerts[:5]


def _deferred_hint(reason: str) -> tuple[str, str]:
    hints = {
        "budget": ("Budget Headroom", "Raise budget or swap out a lower-density project to admit this intervention."),
        "opex": ("OPEX Cap", "Pair this with lower-maintenance projects or relax the operating-cost ceiling."),
        "agency_capacity": ("Agency Capacity", "Shift sequencing or assign support capacity before adding this project."),
        "permit_share": ("Permit Load", "Reduce permit-heavy items or advance pre-clearance before selection."),
        "timeline": ("Timeline Pressure", "Move this into a later phase or widen the target implementation window."),
        "category_concentration": ("Portfolio Concentration", "Use it as a substitute inside the same category rather than stacking more of the same work."),
        "readiness": ("Delivery Readiness", "Complete field design or permit prep before moving it into a fast-delivery portfolio."),
    }
    return hints.get(reason, ("Utility Tradeoff", "Revisit portfolio weights if this project should outrank the current selection."))


def _build_deferred_projects(
    ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    fail_reasons: dict[int, str],
    limit: int = 6,
) -> list[dict[str, object]]:
    selected_ids = {int(item["id"]) for item in selected}
    rows: list[dict[str, object]] = []
    for item in ranked:
        item_id = int(item["id"])
        if item_id in selected_ids:
            continue
        reason = fail_reasons.get(item_id, "utility_tradeoff")
        primary_constraint, action_hint = _deferred_hint(reason)
        rows.append(
            {
                "intervention_id": item_id,
                "title": str(item["title"]),
                "category": str(item["category"]),
                "agency": str(item["agency"]),
                "estimated_cost_lakh": round(float(item["estimated_cost_lakh"]), 3),
                "utility_density": round(float(item.get("utility_density", 0.0)), 3),
                "primary_constraint": primary_constraint,
                "action_hint": action_hint,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _strategy_comparison_row(result: dict[str, Any]) -> dict[str, object]:
    portfolio_summary = result["portfolio_summary"]
    return {
        "strategy": result["strategy_profile"],
        "label": result["strategy_label"],
        "selected_projects": len(result["selected_projects"]),
        "used_budget_lakh": round(float(result["used_budget_lakh"]), 3),
        "impacted_households": int(result["impacted_households"]),
        "estimated_sdg11_gain": round(float(result["estimated_sdg11_gain"]), 3),
        "avg_timeline_months": round(_safe_float(portfolio_summary.get("avg_timeline_months")), 2),
        "permit_share_pct": round(_safe_float(portfolio_summary.get("permit_share_pct")), 2),
        "readiness_score": round(_safe_float(portfolio_summary.get("readiness_score")), 3),
    }


def _simulate_policy_scenario_core(
    items: list[dict[str, Any]],
    budget_lakh: float,
    strategy: str = "balanced",
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    profile = _strategy_profile(strategy, custom_weights)
    enriched = _enrich_items(items, profile)
    ranked = sorted(enriched, key=_priority_score, reverse=True)
    selected, constraints, skipped, fail_reasons = _realworld_select(ranked, budget_lakh, profile)

    used_budget = sum(_safe_float(it.get("estimated_cost_lakh")) for it in selected)
    used_opex = sum(_safe_float(it.get("opex_lakh")) for it in selected)
    permit_count = sum(1 for it in selected if bool(it.get("permit_required")))
    impacted_households = int(sum(_safe_int(it.get("expected_beneficiaries")) for it in selected) / 4.6)
    avg_timeline = float(
        np.mean([_safe_float(it.get("execution_months")) for it in selected]) if selected else 0.0
    )
    utility_sum = sum(_safe_float(it.get("utility_score")) for it in selected)
    climate_projects = sum(1 for it in selected if str(it.get("category")) in CLIMATE_PRIORITY)
    estimated_sdg11_gain = float(np.clip(utility_sum * 6.4 + climate_projects * 0.3, 0.0, 35.0))

    agency_load = _build_agency_load(selected)
    roadmap = _build_roadmap(selected)
    portfolio_summary = _build_portfolio_summary(selected, constraints, roadmap)
    tradeoff_alerts = _build_tradeoff_alerts(
        selected,
        constraints,
        skipped,
        agency_load,
        portfolio_summary,
        profile,
    )
    deferred_projects = _build_deferred_projects(ranked, selected, fail_reasons)

    decision_basis = {
        "strategy_label": profile.label,
        "weight_impact_per_lakh": profile.weights["impact_per_lakh"],
        "weight_equity": profile.weights["equity_need"],
        "weight_urgency": profile.weights["urgency"],
        "weight_feasibility": profile.weights["feasibility"],
        "weight_beneficiaries": profile.weights["beneficiary_norm"],
        "weight_prior_rank": profile.weights["prior_rank_norm"],
        "weight_readiness": profile.weights["readiness_norm"],
        "reserve_lakh": round(constraints.reserve_lakh, 3),
        "investable_budget_lakh": round(constraints.investable_budget_lakh, 3),
        "opex_cap_lakh": round(constraints.opex_cap_lakh, 3),
        "used_opex_lakh": round(used_opex, 3),
        "target_timeline_months": round(constraints.target_timeline_months, 2),
        "avg_selected_timeline_months": round(avg_timeline, 2),
        "agency_capacity_per_agency": constraints.agency_capacity_per_agency,
        "permit_share_cap": round(constraints.permit_share_cap, 3),
        "selected_permit_share": round(permit_count / max(len(selected), 1), 3),
        "selected_count": len(selected),
        "skipped_due_budget": int(skipped.get("budget", 0)),
        "skipped_due_opex": int(skipped.get("opex", 0)),
        "skipped_due_agency_capacity": int(skipped.get("agency_capacity", 0)),
        "skipped_due_permit_share": int(skipped.get("permit_share", 0)),
        "skipped_due_timeline": int(skipped.get("timeline", 0)),
        "skipped_due_concentration": int(skipped.get("category_concentration", 0)),
        "skipped_due_readiness": int(skipped.get("readiness", 0)),
    }

    return {
        "strategy_profile": profile.key,
        "strategy_label": profile.label,
        "strategy_description": profile.description,
        "selected_projects": selected,
        "used_budget_lakh": round(used_budget, 3),
        "remaining_budget_lakh": round(max(0.0, budget_lakh - used_budget), 3),
        "impacted_households": impacted_households,
        "estimated_sdg11_gain": round(estimated_sdg11_gain, 3),
        "selection_method": (
            "strategy-aware constrained utility portfolio "
            "(budget reserve, opex cap, permit share, agency capacity, timeline, category diversity, readiness)"
        ),
        "decision_basis": decision_basis,
        "portfolio_summary": portfolio_summary,
        "selected_reasoning": _selected_reasoning(selected, ranked, profile),
        "agency_load": agency_load,
        "implementation_roadmap": roadmap,
        "tradeoff_alerts": tradeoff_alerts,
        "deferred_projects": deferred_projects,
    }


def simulate_policy_scenario(
    items: list[dict[str, Any]],
    budget_lakh: float,
    strategy: str = "balanced",
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    result = _simulate_policy_scenario_core(items, budget_lakh, strategy, custom_weights)
    result["strategy_comparison"] = build_strategy_comparison(items, budget_lakh)
    return result


def build_counterfactuals(
    items: list[dict[str, Any]],
    base_budget: float,
    strategy: str = "balanced",
    custom_weights: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    scenarios: list[dict[str, float]] = []
    for budget in [base_budget * 0.75, base_budget, base_budget * 1.30]:
        result = _simulate_policy_scenario_core(items, budget, strategy, custom_weights)
        selected = result["selected_projects"]
        permit_share = (
            float(sum(1 for it in selected if bool(it.get("permit_required"))) / max(len(selected), 1))
            if selected
            else 0.0
        )
        avg_timeline = (
            float(np.mean([_safe_float(it.get("execution_months")) for it in selected]))
            if selected
            else 0.0
        )
        scenarios.append(
            {
                "budget_lakh": round(float(budget), 2),
                "used_budget_lakh": round(float(result["used_budget_lakh"]), 2),
                "projects": float(len(selected)),
                "impacted_households": float(int(result["impacted_households"])),
                "estimated_sdg11_gain": round(float(result["estimated_sdg11_gain"]), 2),
                "avg_timeline_months": round(avg_timeline, 2),
                "permit_share_pct": round(permit_share * 100.0, 2),
            }
        )
    return scenarios


def build_strategy_comparison(items: list[dict[str, Any]], budget_lakh: float) -> list[dict[str, object]]:
    rows = [
        _strategy_comparison_row(_simulate_policy_scenario_core(items, budget_lakh, profile.key))
        for profile in STRATEGY_PROFILES.values()
    ]
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("estimated_sdg11_gain")),
            -_safe_float(row.get("readiness_score")),
        )
    )
    return rows


# ---------------------------------------------------------------------------
# AI Budget Sector Prediction
# ---------------------------------------------------------------------------

SECTOR_META: dict[str, dict[str, Any]] = {
    "Drainage": {
        "icon": "🌧️",
        "color": "#3b82f6",
        "rationale_template": "High drain blockage and flood exposure demand urgent drainage investment.",
        "outcome_template": "Reduces flood incidents and waterlogging affecting {hh} households.",
    },
    "Water": {
        "icon": "💧",
        "color": "#06b6d4",
        "rationale_template": "Informal settlements and exposed population indicate acute water access gaps.",
        "outcome_template": "Improves safe water access for {hh} residents in vulnerable zones.",
    },
    "Waste": {
        "icon": "🗑️",
        "color": "#f59e0b",
        "rationale_template": "Dense informal housing correlates with solid waste management deficits.",
        "outcome_template": "Clears waste backlogs and reduces disease vectors for {hh} households.",
    },
    "Road": {
        "icon": "🛣️",
        "color": "#8b5cf6",
        "rationale_template": "Infrastructure gap between household density and road coverage requires repair.",
        "outcome_template": "Restores road connectivity enabling emergency access and daily commutes.",
    },
    "Green": {
        "icon": "🌳",
        "color": "#10b981",
        "rationale_template": "Green deficit index signals a shortfall in tree cover and open public space.",
        "outcome_template": "Adds green buffer reducing urban heat and improving mental well-being.",
    },
    "Public Safety": {
        "icon": "🔒",
        "color": "#ef4444",
        "rationale_template": "Baseline equity and community safety needs justify a minimum safety allocation.",
        "outcome_template": "Extends street lighting and community watch infrastructure across the ward.",
    },
}

_SECTOR_ORDER = list(SECTOR_META.keys())

FLOOR_PCT = 0.04   # every sector gets at least 4%
CEIL_PCT  = 0.44   # no sector exceeds 44%


def _compute_sector_urgency(
    indicators: dict[str, Any], 
    priorities: dict[str, float] = None
) -> dict[str, float]:
    """Convert ward indicator dict into a raw urgency score per sector."""
    # Base indicators...
    blocked_drains     = float(indicators.get("blocked_drain_count", 0) or 0)
    flood_pct          = float(indicators.get("flood_exposure_pct", 0) or 0) / 100.0
    exposed_pop        = float(indicators.get("exposed_population", 0) or 0)
    total_pop          = max(float(indicators.get("total_population", exposed_pop + 1) or 1), 1)
    informal_pct       = float(indicators.get("informal_area_pct", 0) or 0) / 100.0
    green_deficit      = float(indicators.get("green_deficit_index", 0) or 0)
    road_km            = max(float(indicators.get("road_length_km", 1) or 1), 0.1)
    house_count        = float(indicators.get("house_count", 0) or 0)
    equity_need        = float(indicators.get("equity_score", 0.5) or 0.5)

    priorities = priorities or {}

    # Normalise blocked drains to 0-1 (assume 30 drains = high)
    drain_norm  = float(np.clip(blocked_drains / 30.0, 0.0, 1.0))
    pop_norm    = float(np.clip(exposed_pop / max(total_pop, 1), 0.0, 1.0))
    infra_gap   = float(np.clip(house_count / max(road_km * 80.0, 1), 0.0, 1.0))

    scores = {
        "Drainage":      drain_norm * 0.50 + flood_pct * 0.50,
        "Water":         pop_norm   * 0.55 + informal_pct * 0.45,
        "Waste":         informal_pct * 0.65 + drain_norm * 0.35,
        "Road":          infra_gap  * 0.85 + (1.0 - equity_need) * 0.15,
        "Green":         float(np.clip(green_deficit, 0.0, 1.0)) * 0.90 + 0.05,
        "Public Safety": (1.0 - equity_need) * 0.55 + 0.10,
    }

    # Apply AI-driven priorities as a boost factor
    # If a priority is 1.0, it doubles the score. If 0.0, it halves it.
    for k in scores:
        p = priorities.get(k, 0.5) # default to neutral 0.5
        scores[k] = scores[k] * (0.5 + p)

    return scores


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=float)
    # robust softmax
    e = np.exp(vals - vals.max())
    probs = e / e.sum()
    return dict(zip(keys, probs.tolist()))


def _apply_floor_ceil(weights: dict[str, float]) -> dict[str, float]:
    """Enforce per-sector floor and ceiling, renormalise."""
    w = dict(weights)
    # Apply floor
    for k in w:
        if w[k] < FLOOR_PCT:
            w[k] = FLOOR_PCT
    # Normalise
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    # Apply ceiling
    capped = False
    for k in w:
        if w[k] > CEIL_PCT:
            w[k] = CEIL_PCT
            capped = True
    if capped:
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
    return w


def predict_budget_allocation(
    indicators: dict[str, Any],
    budget_lakh: float,
    sector_priorities: dict[str, float] = None,
    sector_rationales: dict[str, str] = None,
) -> dict[str, Any]:
    """
    Given ward indicators and a total budget in lakh BDT, predict the  
    optimal sector-wise allocation using urgency scoring + softmax weighting.
    """
    budget_lakh = max(float(budget_lakh), 0.5)

    urgency_raw   = _compute_sector_urgency(indicators, priorities=sector_priorities)
    softmax_w     = _softmax(urgency_raw)
    final_weights = _apply_floor_ceil(softmax_w)

    # Estimate total beneficiary households across all sectors
    exposed_pop   = float(indicators.get("exposed_population", 1000) or 1000)
    total_hh      = max(int(exposed_pop / 4.6), 1)

    # Build sector rows
    sectors: list[dict[str, Any]] = []
    for sector in _SECTOR_ORDER:
        pct    = round(final_weights[sector] * 100.0, 1)
        amount = round(final_weights[sector] * budget_lakh, 2)
        meta   = SECTOR_META[sector]
        hh_impact = max(1, int(total_hh * final_weights[sector] * 1.4))
        # Use AI-generated rationale if available, otherwise template
        rationale = (sector_rationales or {}).get(sector, meta["rationale_template"])
        
        sectors.append({
            "name":            sector,
            "icon":            meta["icon"],
            "color":           meta["color"],
            "allocation_pct":  pct,
            "allocation_lakh": amount,
            "urgency_score":   round(urgency_raw[sector], 4),
            "rationale":       rationale,
            "expected_outcome": meta["outcome_template"].format(hh=f"{hh_impact:,}"),
        })

    # Sort by allocation descending for display
    sectors.sort(key=lambda s: s["allocation_lakh"], reverse=True)

    top_sector    = sectors[0]["name"]
    second_sector = sectors[1]["name"] if len(sectors) > 1 else ""
    estimated_sdg = round(float(np.clip(budget_lakh * 0.18, 0.5, 35.0)), 2)

    ai_summary = (
        f"Based on this ward's live indicators, the AI recommends prioritising "
        f"{top_sector.lower()} ({sectors[0]['allocation_pct']}%) and "
        f"{second_sector.lower()} ({sectors[1]['allocation_pct'] if second_sector else 0}%) "
        f"to maximise urban resilience outcomes from a BDT {budget_lakh:.1f} lakh budget. "
        f"Projected SDG-11 gain: +{estimated_sdg} points."
    )

    return {
        "ward_budget_lakh":    budget_lakh,
        "sectors":             sectors,
        "ai_summary":          ai_summary,
        "estimated_sdg11_gain": estimated_sdg,
        "projected_households": total_hh,
        "top_sector":          top_sector,
    }
