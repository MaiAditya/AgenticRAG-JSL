from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import document_router
from src.core.error_handling import setup_error_handling
from src.core.config import settings

app = FastAPI(title="Document Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router)

# Setup error handling
setup_error_handling(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 