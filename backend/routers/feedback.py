from fastapi import APIRouter
from pydantic import BaseModel
from backend.services import repo

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

class FeedbackIn(BaseModel):
    article_id: str
    feedback_type: str  # 'like' or 'dislike'
    user_id: str = 'guest'

@router.post("")
def create_feedback(feedback: FeedbackIn):
    result = repo.add_feedback(feedback.article_id, feedback.user_id, feedback.feedback_type)
    return result

@router.get("")
def get_feedback():
    df = repo.get_all_feedback()
    if df.empty:
        return []
    return df.to_dict(orient="records")