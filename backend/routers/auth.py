from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator
from backend.services import repo

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email address")
        return normalized


class DeleteRequest(BaseModel):
    user_id: str
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


@router.get("/me")
def get_me(user_id: str = Query(...)):
    user = repo.get_user_info(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/me")
def delete_me(body: DeleteRequest):
    try:
        repo.authenticate_user(body.user_id, body.password)
    except ValueError:
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")
    deleted = repo.delete_user(body.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}
