from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def export_worklist_csv(output_path: Path, items: list[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(items)
    frame.to_csv(output_path, index=False)
    return output_path


def export_policy_pdf(
    output_path: Path,
    ward_name: str,
    budget_lakh: float,
    strategy_label: str,
    strategy_description: str,
    selected_items: list[dict[str, Any]],
    counterfactuals: list[dict[str, float]],
    portfolio_summary: dict[str, Any] | None = None,
    roadmap: list[dict[str, Any]] | None = None,
    tradeoff_alerts: list[dict[str, Any]] | None = None,
    ai_budget_plan: dict[str, Any] | None = None,
    indicators: dict[str, Any] | None = None,
) -> Path:
    portfolio_summary = portfolio_summary or {}
    roadmap = roadmap or []
    tradeoff_alerts = tradeoff_alerts or []
    ai_budget_plan = ai_budget_plan or {}
    indicators = indicators or {}

    def ensure_space(current_y: float, required: float = 22) -> float:
        if current_y >= required:
            return current_y
        pdf.showPage()
        pdf.setFont("Helvetica", 9)
        return height - 40

    def draw_lines(lines: list[str], current_y: float, step: float = 12) -> float:
        for line in lines:
            current_y = ensure_space(current_y, 25)
            pdf.drawString(42, current_y, line[:118])
            current_y -= step
        return current_y

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    y = height - 45
    
    # --- Header ---
    pdf.setTitle(f"UrbanSightAI Policy Memo - {ward_name}")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "UrbanSightAI Policy Memo")
    y -= 24
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Ward: {ward_name}")
    y -= 14
    pdf.drawString(40, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    y -= 14
    pdf.drawString(40, y, f"Planning Strategy: {strategy_label}")
    y -= 14
    pdf.drawString(40, y, f"Reference Budget: {budget_lakh:.2f} lakh BDT")
    y -= 25

    # --- 1. Risk & Indicators ---
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "I. Neighborhood Risk & Indicators")
    y -= 18
    pdf.setFont("Helvetica", 9)
    risk_lines = [
        f"Flood Risk Index: {indicators.get('flood_risk', 0.0)*100:.1f} % | Blocked Drains: {indicators.get('blocked_drains', 0)} units",
        f"Informal Area: {indicators.get('informal_area', 0.0):.1f} % | Green Deficit: {indicators.get('green_deficit', 0.0):.3f}",
        f"Exposed Population: {indicators.get('exposed_pop', 0):,} people | SDG-11 Baseline: {indicators.get('sdg11_score', 0.0):.2f}",
    ]
    y = draw_lines(risk_lines, y)
    
    # Neighborhood Problems (derived from indicators)
    y -= 6
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(42, y, "Active Neighborhood Problems:")
    y -= 14
    pdf.setFont("Helvetica", 9)
    problems = []
    if indicators.get('blocked_drains', 0) > 40:
        problems.append("CRITICAL: Massive drainage blockage network detected (80+ segments).")
    if indicators.get('flood_risk', 0.0) > 0.25:
        problems.append("CRITICAL: High surface flood exposure in current ward profile.")
    if indicators.get('green_deficit', 0.0) > 0.5:
        problems.append("MEDIUM: Substantial green space and tree cover deficit.")
    if not problems:
        problems.append("Monitor: No critical outliers detected in current indicator set.")
    y = draw_lines(problems[:3], y)
    y -= 15

    # --- 2. AI Budget Sector Distribution ---
    y = ensure_space(y, 150)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "II. AI Budget Sector Distribution (Interactive Prediction)")
    y -= 18
    pdf.setFont("Helvetica", 9)
    if ai_budget_plan.get("sectors"):
        for sector in ai_budget_plan["sectors"]:
            line = f"- {sector['name']}: {sector['allocation_pct']:.1f}% ({sector['allocation_lakh']:.2f} lakh)"
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(45, y, line)
            y -= 12
            pdf.setFont("Helvetica-Oblique", 8)
            pdf.drawString(60, y, f"Rationale: {sector.get('rationale', 'Standard allocation')}")
            y -= 10
            y = ensure_space(y, 40)
    else:
        pdf.drawString(42, y, "No AI sector-wise allocation available for this scenario.")
        y -= 12
        
    if ai_budget_plan.get("ai_summary"):
        y -= 5
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(42, y, "AI Summary Strategy:")
        y -= 13
        pdf.setFont("Helvetica", 8.5)
        narrative = ai_budget_plan["ai_summary"]
        # Simple word wrap for narrative
        words = narrative.split()
        line = ""
        for word in words:
            if len(line + " " + word) < 110:
                line += " " + word
            else:
                pdf.drawString(45, y, line.strip())
                y -= 11
                line = word
                y = ensure_space(y, 30)
        pdf.drawString(45, y, line.strip())
        y -= 20

    # --- 3. AI Policy Prioritization ---
    y = ensure_space(y, 100)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "III. AI Policy Prioritization (Optimal Worklist)")
    y -= 18
    pdf.setFont("Helvetica", 9)
    for idx, item in enumerate(selected_items[:8], start=1):
        line = (
            f"#{idx} {item['title']} | {item['category']} | "
            f"Cost {item['estimated_cost_lakh']:.2f}L | Beneficiaries {item['expected_beneficiaries']}"
        )
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(42, y, line)
        y -= 12
        justification = item.get('justification', f"Priority weighted by {item.get('ranking_score', 0.0)}")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(55, y, f"Justification: {justification}")
        y -= 11
        y = ensure_space(y, 40)

    # --- 4. Execution & Roadmap ---
    if roadmap:
        y -= 10
        y = ensure_space(y, 100)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "IV. Phased Delivery Roadmap")
        y -= 18
        pdf.setFont("Helvetica", 9)
        
        for step in roadmap[:6]:
            line = (
                f"{step['phase']}: {step['title']} | {step['agency']} | "
                f"Timeline: Month {int(step['start_month'])}-{int(step['end_month'])}"
            )
            pdf.drawString(42, y, line)
            y -= 12
            y = ensure_space(y, 40)

    # Alerts & Scenarios
    if tradeoff_alerts:
        y -= 10
        y = ensure_space(y, 90)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(40, y, "Operational Tradeoff Alerts:")
        y -= 15
        pdf.setFont("Helvetica", 8.5)
        for alert in tradeoff_alerts[:3]:
            pdf.drawString(45, y, f"* {alert['severity'].upper()} | {alert['topic']}: {alert['message']}")
            y -= 11

    y -= 15
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y, "Investment Impact Summary")
    y -= 16
    pdf.setFont("Helvetica", 9)
    summary_txt = f"Total Budget: {budget_lakh:.1f} Lakh | Households Filtered: ~{ai_budget_plan.get('projected_households', 0):,}"
    pdf.drawString(42, y, summary_txt)
    y -= 12
    gain_txt = f"Projected Ward SDG-11 Gain: +{ai_budget_plan.get('estimated_sdg11_gain', 0.0):.2f} pts"
    pdf.drawString(42, y, gain_txt)
    y -= 15

    # Footer checklist
    y = ensure_space(y, 80)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, "Goverance Next Steps:")
    y -= 14
    pdf.setFont("Helvetica", 8)
    pdf.drawString(45, y, "1. Verify field evidence and geospatial layer confidence.")
    y -= 10
    pdf.drawString(45, y, "2. Confirm lead agency + supporting agency assignment.")
    y -= 10
    pdf.drawString(45, y, "3. Complete permit screening and procurement mode selection.")
    y -= 10
    pdf.drawString(45, y, "4. Issue work order with timeline milestones and log audit event.")
    
    pdf.save()
    return output_path
