from fastapi import APIRouter, Query

from backend.schemas import ArticleOut
from backend.services import recommend


router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("", response_model=list[ArticleOut])
def get_recommendations(
    session_id: str | None = Query(None),
    user_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    return recommend.get_recommendations(session_id=session_id, user_id=user_id, limit=limit)
