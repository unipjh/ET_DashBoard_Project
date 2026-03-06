import os
from dotenv import load_dotenv


def get_gemini_api_key() -> str:
    # 1순위: 로컬 .env
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    # 2순위: Streamlit Cloud secrets
    if not key:
        try:
            import streamlit as st
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    return key or ""
