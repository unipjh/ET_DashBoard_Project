# backend/services/process_status.py
import uuid

STATUS = {
    "process_name": "idle",
    "current_step": "대기 중",
    "last_message": "",
    "current_job_id": None,
}


def start_job(job_type: str) -> str:
    """배경 작업 시작: in-memory 상태 갱신 + DB INSERT."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    STATUS["current_job_id"] = job_id
    STATUS["process_name"] = job_type
    try:
        from backend.services import repo
        repo.insert_background_job(job_id, job_type)
    except Exception:
        pass
    return job_id


def update_status(step: str, message: str):
    """단계·메시지 갱신: in-memory + DB (비차단)."""
    STATUS["current_step"] = step
    STATUS["last_message"] = message
    print(message)
    job_id = STATUS.get("current_job_id")
    if job_id:
        try:
            from backend.services import repo
            repo.update_background_job_step(job_id, step, message)
        except Exception:
            pass


def finish_job(articles_processed: int = 0):
    """작업 성공 완료: DB를 'done'으로 갱신."""
    job_id = STATUS.get("current_job_id")
    if job_id:
        try:
            from backend.services import repo
            repo.finish_background_job(job_id, articles_processed)
        except Exception:
            pass


def fail_job(error_text: str):
    """작업 실패: DB를 'failed'로 갱신."""
    job_id = STATUS.get("current_job_id")
    if job_id:
        try:
            from backend.services import repo
            repo.fail_background_job(job_id, error_text)
        except Exception:
            pass


def reset_status():
    """in-memory 상태를 idle로 초기화."""
    STATUS["process_name"] = "idle"
    STATUS["current_step"] = "대기 중"
    STATUS["last_message"] = ""
    STATUS["current_job_id"] = None
