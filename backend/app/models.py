from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="viewer")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Ward(Base):
    __tablename__ = "wards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    area_km2: Mapped[float] = mapped_column(Float)
    population: Mapped[int] = mapped_column(Integer)
    households: Mapped[int] = mapped_column(Integer)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    bbox_json: Mapped[str] = mapped_column(Text, default="[]")

    indicators: Mapped[list["WardIndicator"]] = relationship(back_populates="ward")
    interventions: Mapped[list["Intervention"]] = relationship(back_populates="ward")


class WardIndicator(Base):
    __tablename__ = "ward_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ward_id: Mapped[int] = mapped_column(ForeignKey("wards.id"), index=True)
    informal_area_pct: Mapped[float] = mapped_column(Float)
    blocked_drain_count: Mapped[int] = mapped_column(Integer)
    green_deficit_index: Mapped[float] = mapped_column(Float)
    flood_risk_index: Mapped[float] = mapped_column(Float)
    sdg11_score: Mapped[float] = mapped_column(Float)
    exposed_population: Mapped[int] = mapped_column(Integer)

    ward: Mapped[Ward] = relationship(back_populates="indicators")


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ward_id: Mapped[int] = mapped_column(ForeignKey("wards.id"), index=True)
    title: Mapped[str] = mapped_column(String(140))
    category: Mapped[str] = mapped_column(String(60))
    agency: Mapped[str] = mapped_column(String(60))
    permit_required: Mapped[bool] = mapped_column(Boolean, default=False)
    estimated_cost_lakh: Mapped[float] = mapped_column(Float)
    expected_beneficiaries: Mapped[int] = mapped_column(Integer)
    beneficiary_ci_low: Mapped[int] = mapped_column(Integer, default=0)
    beneficiary_ci_high: Mapped[int] = mapped_column(Integer, default=0)
    beneficiary_method: Mapped[str] = mapped_column(String(120), default="statistical-inference-v1")
    impact_per_lakh: Mapped[float] = mapped_column(Float, default=0.0)
    feasibility: Mapped[float] = mapped_column(Float)
    equity_need: Mapped[float] = mapped_column(Float)
    urgency: Mapped[float] = mapped_column(Float)
    ranking_score: Mapped[float] = mapped_column(Float, default=0.0)
    justification: Mapped[str] = mapped_column(Text, default="")

    ward: Mapped[Ward] = relationship(back_populates="interventions")


class CitizenReport(Base):
    __tablename__ = "citizen_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ward_id: Mapped[int] = mapped_column(ForeignKey("wards.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(20), default="bangla")
    category: Mapped[str] = mapped_column(String(50))
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    priority_weight: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    actor_username: Mapped[str] = mapped_column(String(50), default="system")
    actor_role: Mapped[str] = mapped_column(String(20), default="system")
    action: Mapped[str] = mapped_column(String(80), index=True)
    ward_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
