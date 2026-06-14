from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from backend.services import repo

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup")
def signup(body: AuthRequest):
    try:
        user = repo.create_user(body.email, body.password)
        return {"ok": True, "user_id": user["user_id"]}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/login")
def login(body: AuthRequest):
    try:
        user = repo.authenticate_user(body.email, body.password)
        return {"ok": True, "user_id": user["user_id"]}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
