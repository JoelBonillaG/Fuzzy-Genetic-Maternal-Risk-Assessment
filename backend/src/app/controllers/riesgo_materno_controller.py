from fastapi import APIRouter, HTTPException

from ..schemas.prediccion import (
    AjusteEntradaResponse,
    ExplicacionResponse,
    MembresiasResponse,
    PrediccionRequest,
    PrediccionResponse,
)
from ..services.riesgo_materno_service import (
    explicar_prediccion,
    obtener_membresias,
    predecir_riesgo_materno,
)


router = APIRouter(prefix="/api/v1/predicciones", tags=["Predicciones"])


@router.get("/membresias", response_model=MembresiasResponse)
def obtener_membresias_endpoint() -> MembresiasResponse:
    """Curvas de membresia base para graficar."""
    resultado = obtener_membresias()
    return MembresiasResponse(**resultado)


@router.post("/riesgo-materno/explicacion", response_model=ExplicacionResponse)
def explicar_prediccion_endpoint(payload: PrediccionRequest) -> ExplicacionResponse:
    """Explicacion detallada de una prediccion."""
    try:
        resultado = explicar_prediccion(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplicacionResponse(**resultado)


@router.post("/riesgo-materno", response_model=PrediccionResponse)
def predecir_riesgo_materno_endpoint(payload: PrediccionRequest) -> PrediccionResponse:
    """Prediccion simple del riesgo materno."""
    try:
        resultado = predecir_riesgo_materno(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PrediccionResponse(
        puntaje=resultado["puntaje"],
        riesgo=resultado["riesgo"],
        sin_activacion=resultado["sin_activacion"],
        sistema=resultado["sistema"],
        origen_modelo=resultado["origen_modelo"],
        cantidad_reglas_activas=resultado["cantidad_reglas_activas"],
        ajustes_entrada=[
            AjusteEntradaResponse(**ajuste)
            for ajuste in resultado["ajustes_entrada"]
        ],
    )
