import uvicorn

from .core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
