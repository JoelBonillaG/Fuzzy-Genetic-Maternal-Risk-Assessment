"""Endpoint de solo lectura: expone la seleccion de reglas guardada por el CLI.

El entrenamiento se ejecuta desde:
    python -m src.riesgo_materno.herramientas.entrenar_ag
"""

from fastapi import APIRouter

from ..schemas.prediccion import SeleccionReglasResponse
from ..services.riesgo_materno_service import obtener_seleccion_reglas_actual

router = APIRouter(prefix="/api/v1/ga", tags=["Algoritmo genetico"])


@router.get("/seleccion-reglas", response_model=SeleccionReglasResponse)
def seleccion_reglas() -> SeleccionReglasResponse:
    """Devuelve la base de reglas optimizada actualmente persistida."""
    return SeleccionReglasResponse(**obtener_seleccion_reglas_actual())
