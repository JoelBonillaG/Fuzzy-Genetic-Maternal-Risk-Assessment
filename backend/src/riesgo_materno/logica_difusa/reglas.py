# Reglas del sistema difuso Mamdani para riesgo materno.
#
# Carga las reglas aprendidas por RIPPER desde reglas_aprendidas.json.
# Para regenerar las reglas: python -m src.riesgo_materno.entrenamiento.ripper

import json

from ..entrenamiento.modelo import RUTA_REGLAS_APRENDIDAS


def _leer_json(ruta):
    """Lee un JSON de reglas y convierte antecedentes a tuplas."""
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    for r in contenido:
        r["antecedentes"] = [tuple(ant) for ant in r["antecedentes"]]
    return contenido


def _cargar_reglas():
    """Carga reglas aprendidas"""
    if not RUTA_REGLAS_APRENDIDAS.exists():
        raise FileNotFoundError(
            f"No se encontraron reglas aprendidas en {RUTA_REGLAS_APRENDIDAS}."
        )
    return _leer_json(RUTA_REGLAS_APRENDIDAS)


# REGLAS es lo que consume el resto del sistema (motor.py, service.py).
REGLAS = _cargar_reglas()
