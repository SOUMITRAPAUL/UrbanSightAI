from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditEvent, User


def log_audit_event(
    db: Session,
    action: str,
    user: User | None = None,
    ward_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    event = AuditEvent(
        actor_username=user.username if user else "system",
        actor_role=user.role if user else "system",
        action=action,
        ward_id=ward_id,
        details_json=json.dumps(details or {}, ensure_ascii=True),
    )
    db.add(event)
    db.commit()


def list_audit_events(
    db: Session,
    ward_id: int | None = None,
    limit: int = 40,
) -> list[AuditEvent]:
    stmt = select(AuditEvent).order_by(AuditEvent.timestamp.desc()).limit(limit)
    if ward_id is not None:
        stmt = (
            select(AuditEvent)
            .where(AuditEvent.ward_id == ward_id)
            .order_by(AuditEvent.timestamp.desc())
            .limit(limit)
        )
    return db.scalars(stmt).all()

