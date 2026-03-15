from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import articles, search, admin, trust
from backend.services import repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo.init_db()
    yield


app = FastAPI(title="ET API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(articles.router)
app.include_router(search.router)
app.include_router(admin.router)
app.include_router(trust.router)


@app.get("/")
def root():
    return {"message": "ET API is running"}
