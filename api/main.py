"""
FastAPI Main Application - Multi-RAG Agent System API
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from api.routes import router
from config.settings import settings
from src.utils import api_logger

# Configure logging
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    
    # Startup
    api_logger.info("Starting Multi-RAG Agent System API")
    api_logger.info(f"Settings: {settings.default_llm_model}")
    
    yield
    
    # Shutdown
    api_logger.info("Shutting down Multi-RAG Agent System API")


# Create FastAPI app
app = FastAPI(
    title="Multi-RAG Agent System",
    description="Multi-agent Retrieval-Augmented Generation system using Google's Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Include routers
app.include_router(router)

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=frontend_dist, html=True), name="frontend")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-RAG Agent System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    api_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        reload=False
    )
