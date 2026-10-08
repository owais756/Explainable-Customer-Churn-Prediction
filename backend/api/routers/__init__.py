# backend/api/routers/__init__.py
from .customers import router as customers_router
from .analytics import router as analytics_router
from .predictions import router as predictions_router

__all__ = ["customers_router", "analytics_router", "predictions_router"]
