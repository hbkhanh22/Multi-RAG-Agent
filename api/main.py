"""
FastAPI Main Application - Multi-RAG Agent System API
"""

import logging
import threading
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
from src.ingestion import auto_ingest_raw_documents
from src.mcp_client import mcp_manager

# Configure logging
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    
    # Startup
    api_logger.info("Starting Multi-RAG Agent System API")
    api_logger.info(f"Settings: {settings.default_llm_model}")

    def _ingest():
        try:
            auto_ingest_raw_documents()
        except Exception as e:
            api_logger.error(f"Error during auto-ingestion: {e}", exc_info=True)

    # Run ingestion in the background so the API starts answering immediately
    threading.Thread(target=_ingest, name="auto-ingest", daemon=True).start()

    # Connect to MCP servers in the background too (npx/uvx may download packages on first run)
    mcp_manager.start()

    yield
    
    # Shutdown
    mcp_manager.stop()
    api_logger.info("Shutting down Multi-RAG Agent System API")


# Create FastAPI app
app = FastAPI(
    title="Multi-RAG Agent System",
    description="Multi-agent Retrieval-Augmented Generation system (LangGraph + Chroma + NVIDIA NIM)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
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
    api_logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred.",
            "path": request.url.path
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
