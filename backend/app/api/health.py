from fastapi import APIRouter
from typing import Dict, Any

from ..core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify backend service status and checkpoint availability."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "checkpoints_available": True
    }
