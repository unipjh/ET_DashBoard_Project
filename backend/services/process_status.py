# backend/services/process_status.py

STATUS = {
    "process_name": "idle",      # "idle", "crawl", "analyze", etc.
    "current_step": "대기 중",    # "crawl.py", "admin_pipeline.py", "repo.py"
    "last_message": "",          # The log message from print()
}

def update_status(step: str, message: str):
    """Updates the global process status."""
    STATUS["current_step"] = step
    STATUS["last_message"] = message
    # Also print to console for backend logging
    print(message)

def reset_status():
    """Resets the status to idle."""
    STATUS["process_name"] = "idle"
    STATUS["current_step"] = "대기 중"
    STATUS["last_message"] = ""