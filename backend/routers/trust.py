from fastapi import APIRouter, HTTPException
from backend.services import repo

router = APIRouter(prefix="/api/trust", tags=["trust"])


@router.get("/{article_id}")
def get_trust(article_id: str):
    df = repo.load_articles()
    row = df[df["article_id"] == article_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    r = row.iloc[0]
    return {
        "article_id": article_id,
        "trust_score": int(r.get("trust_score", 0) or 0),
        "trust_verdict": str(r.get("trust_verdict", "") or ""),
        "trust_reason": str(r.get("trust_reason", "") or ""),
        "trust_per_criteria": str(r.get("trust_per_criteria", "") or ""),
    }
