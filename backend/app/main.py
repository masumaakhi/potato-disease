from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api import health, models, predict

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for An Attention-Guided Lightweight CNN–ViT Fusion Network for Potato Leaf Disease Classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Next.js frontend communication
cors_origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registration
app.include_router(health.router)                                 # GET /health
app.include_router(health.router, prefix=settings.API_V1_STR)     # GET /api/health
app.include_router(models.router, prefix=settings.API_V1_STR)     # GET /api/models, GET /api/models/{id}
app.include_router(predict.router, prefix=settings.API_V1_STR)    # POST /api/predict, GET /api/predict/status


@app.get("/")
def root():
    """Service root providing discovery information."""
    return {
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "models": f"{settings.API_V1_STR}/models",
            "predict": f"{settings.API_V1_STR}/predict",
            "predict_status": f"{settings.API_V1_STR}/predict/status",
            "documentation": "/docs"
        }
    }
