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


@app.on_event("startup")
def startup_optimization():
    """Optimizes CPU thread allocation and warms up the primary research model."""
    import torch
    # Crucial for cloud containers (Render/Docker) to prevent CPU thread thrashing
    torch.set_num_threads(2)

    # Preload the primary proposed architecture into memory
    try:
        from .services.inference import inference_service
        inference_service.get_or_load_model("attention_fusion", seed=42)
    except Exception as exc:
        print(f"[WARMUP] Could not pre-warm model: {exc}")


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
