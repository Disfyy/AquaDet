from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.routes import router

app = FastAPI(title="AquaDet API", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
