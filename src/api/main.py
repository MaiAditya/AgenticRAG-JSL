from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import document, query, stats
from src.core.error_handling import setup_error_handling

app = FastAPI(
    title="CPG Assistant API",
    description="API for CPG document processing and querying",
    version="0.1.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handling
setup_error_handling()

# Include routers
app.include_router(document.router)
app.include_router(query.router)
app.include_router(stats.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 