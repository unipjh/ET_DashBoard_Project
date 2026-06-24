import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import articles, search, admin, trust, feedback, logs, stocks, auth, recommendations
from backend.services import repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_startup_error = None
    try:
        repo.init_db()
    except Exception as e:
        app.state.db_startup_error = f"{type(e).__name__}: {e}"
        print(f"[STARTUP WARN] DB initialization failed: {app.state.db_startup_error}")
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


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["Server-Timing"] = f"app;dur={duration_ms:.1f}"
    print(f"[HTTP] {request.method} {request.url.path} {response.status_code} {duration_ms:.1f}ms")
    return response

app.include_router(articles.router)
app.include_router(search.router)
app.include_router(admin.router)
app.include_router(trust.router)
app.include_router(feedback.router)
app.include_router(logs.router)
app.include_router(stocks.router)
app.include_router(auth.router)
app.include_router(recommendations.router)


@app.get("/")
def root():
    return {"message": "ET API is running"}


@app.get("/health")
def health():
    ok, error = repo.check_db()
    if ok:
        return {"status": "ok", "database": "ok"}

    detail = getattr(app.state, "db_startup_error", None) or error
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "database": "unavailable", "detail": detail},
    )
