from fastapi import APIRouter

from ..schemas.prediccion import FuzzyDefinicionesResponse, FuzzyReglasResponse
from ..services.riesgo_materno_service import obtener_definiciones_difusas, obtener_reglas_difusas

router = APIRouter(prefix="/api/v1/difuso", tags=["Logica difusa"])


@router.get("/definiciones", response_model=FuzzyDefinicionesResponse)
def definiciones_difusas() -> FuzzyDefinicionesResponse:
    return FuzzyDefinicionesResponse(**obtener_definiciones_difusas())


@router.get("/reglas", response_model=FuzzyReglasResponse)
def reglas_difusas() -> FuzzyReglasResponse:
    return FuzzyReglasResponse(**obtener_reglas_difusas())
