from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .controllers.difuso_controller import router as difuso_router
from .controllers.ga_controller import router as ga_router
from .controllers.riesgo_materno_controller import router as riesgo_materno_router
from .core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(riesgo_materno_router)
app.include_router(ga_router)
app.include_router(difuso_router)


@app.get("/health")
def health():
    return {"status": "ok"}
