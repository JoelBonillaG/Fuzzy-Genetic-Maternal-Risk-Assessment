from fastapi import APIRouter, HTTPException

from ..schemas.prediccion import (
    AjusteEntradaResponse,
    ExplicacionResponse,
    MembresiasResponse,
    PrediccionRequest,
    PrediccionResponse,
)
from ..services.riesgo_materno_service import explicar_prediccion, obtener_membresias, predecir_riesgo_materno


router = APIRouter(prefix="/api/v1/predicciones", tags=["Predicciones"])

"""Curvas de membresia para graficar. Se combina este primero para dibujar las curvas."""
@router.get("/membresias", response_model=MembresiasResponse)
def obtener_membresias_endpoint() -> MembresiasResponse:
    resultado = obtener_membresias()
    return MembresiasResponse(**resultado)

"""Explicacion de la prediccion. Se llama a este endpoint una vez se hayan dibujado las curvas."""
@router.post("/riesgo-materno/explicacion", response_model=ExplicacionResponse)
def explicar_prediccion_endpoint(payload: PrediccionRequest) -> ExplicacionResponse:
    try:
        resultado = explicar_prediccion(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplicacionResponse(**resultado)

"""Prediccion simple."""
@router.post("/riesgo-materno", response_model=PrediccionResponse)
def predecir_riesgo_materno_endpoint(payload: PrediccionRequest) -> PrediccionResponse:
    try:
        resultado = predecir_riesgo_materno(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PrediccionResponse(
        puntaje=resultado["puntaje"],
        riesgo=resultado["riesgo"],
        sistema=resultado["sistema"],
        origen_modelo=resultado["origen_modelo"],
        ajustes_entrada=[
            AjusteEntradaResponse(**ajuste)
            for ajuste in resultado["ajustes_entrada"]
        ],
    )
