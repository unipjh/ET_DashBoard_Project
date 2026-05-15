import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import articles, search, admin, trust, feedback
from backend.services import repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo.init_db()
    yield


app = FastAPI(title="ET API", version="1.0.0", lifespan=lifespan)

# ALLOWED_ORIGINS 환경변수로 허용 도메인 지정 (쉼표 구분)
# 예: "https://your-app.vercel.app,http://localhost:5173"
_raw = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw.split(",")] if _raw != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(articles.router)
app.include_router(search.router)
app.include_router(admin.router)
app.include_router(trust.router)
app.include_router(feedback.router)


@app.get("/")
def root():
    return {"message": "ET API is running"}
