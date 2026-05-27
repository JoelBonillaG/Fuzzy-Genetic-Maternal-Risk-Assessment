# Reglas del sistema difuso Mamdani para riesgo materno.
#
# Carga las reglas publicadas para el sistema web.
# Para publicar reglas limpias:
# python -m src.riesgo_materno.pipeline_reglas.preparar_reglas_web

import json

from ..entrenamiento.modelo import (
    RUTA_REGLAS_SISTEMA_DIFUSO,
    RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER,
)


def _leer_json(ruta):
    """Lee un JSON de reglas y convierte antecedentes a tuplas."""
    contenido = json.loads(ruta.read_text(encoding="utf-8-sig"))
    for r in contenido:
        r["antecedentes"] = [tuple(ant) for ant in r["antecedentes"]]
    return contenido


def cargar_reglas_desde_ruta(ruta):
    """Carga reglas Mamdani desde una ruta JSON concreta."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontraron reglas en {ruta}.")
    return _leer_json(ruta)


def _cargar_reglas():
    """Carga las reglas publicadas para el sistema difuso."""
    ruta = RUTA_REGLAS_SISTEMA_DIFUSO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontraron reglas publicadas en {RUTA_REGLAS_SISTEMA_DIFUSO}."
        )
    return _leer_json(ruta)


# REGLAS es lo que consume el resto del sistema (motor.py, service.py).
REGLAS = _cargar_reglas()
REGLAS_RIPPER = cargar_reglas_desde_ruta(RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER)
