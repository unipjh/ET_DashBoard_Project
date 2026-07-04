from fastapi import APIRouter, HTTPException
from backend.services import repo

router = APIRouter(prefix="/api/trust", tags=["trust"])


@router.get("/{article_id}")
def get_trust(article_id: str):
    # DB에서 특정 기사의 신뢰도 정보만 직접 조회합니다.
    trust_data = repo.get_trust_by_article_id(article_id)
    if not trust_data:
        raise HTTPException(status_code=404, detail="Article not found")
    return trust_data
