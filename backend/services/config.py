import os
from dotenv import load_dotenv


def get_gemini_api_key() -> str:
    load_dotenv()
    return os.getenv("GEMINI_API_KEY") or ""


def get_database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")
    return url


def get_admin_password() -> str:
    load_dotenv()
    return os.getenv("ADMIN_PASSWORD") or ""
