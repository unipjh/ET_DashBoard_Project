from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.services import repo

router = APIRouter(prefix="/api/logs", tags=["logs"])

class LogEvent(BaseModel):
    session_id: str
    event_type: str
    article_id: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

@router.post("")
def record_log(event: LogEvent, background_tasks: BackgroundTasks):
    payload = event.model_dump()
    if event.event_type == "impression":
        background_tasks.add_task(repo.insert_impression_log, payload)
    else:
        background_tasks.add_task(repo.insert_log, payload)
    return {"status": "success"}
