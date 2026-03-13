from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db import get_db
from app.models import User
from app.schemas import TokenResponse, UserLogin, UserProfile, UserRegister
from app.services.audit import log_audit_event


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserProfile)
def register(payload: UserRegister, db: Session = Depends(get_db)) -> UserProfile:
    existing = db.scalar(select(User).where(User.username == payload.username))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is taken.")
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    log_audit_event(
        db,
        action="user_registered",
        user=user,
        details={"role": user.role},
    )
    return UserProfile(username=user.username, role=user.role)


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.username == payload.username))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    token = create_access_token({"sub": user.username, "role": user.role})
    log_audit_event(
        db,
        action="user_login",
        user=user,
        details={},
    )
    return TokenResponse(access_token=token, username=user.username, role=user.role)


@router.get("/me", response_model=UserProfile)
def me(user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile(username=user.username, role=user.role)
