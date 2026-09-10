from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
