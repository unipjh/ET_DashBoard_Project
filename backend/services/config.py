import os
from dotenv import load_dotenv


def get_gemini_api_key() -> str:
    load_dotenv()
    return os.getenv("GEMINI_API_KEY") or ""
