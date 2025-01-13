from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import document_router, query_router, stats_router
from src.core.error_handling import setup_error_handling
from src.core.config import settings
from src.core.initialization import initialize_models
from loguru import logger

app = FastAPI(
    title="Document Processing API",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Initialize models during application startup"""
    logger.info("Starting application initialization...")
    try:
        # Verify settings are loaded
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        logger.info("Environment variables verified")
        
        # Initialize models
        initialize_models()
        logger.info("Application initialization complete!")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router)
app.include_router(query_router)
app.include_router(stats_router)

# Setup error handling
setup_error_handling(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 