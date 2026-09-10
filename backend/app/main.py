from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.reports import router as reports_router
from app.api.endpoints.studies import router as studies_router

app = FastAPI(
    title="Agentic Knee MRI API",
    description="Backend for the Agentic Knee MRI Analysis & Reporting System",
    version="1.0.0"
)

# CORS setup for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(studies_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Agentic Knee MRI Backend"}

# Serve frontend build in production / single-container deployment
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
