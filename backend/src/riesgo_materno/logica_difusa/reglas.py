# Reglas del sistema difuso Mamdani para riesgo materno.
#
# Carga las reglas publicadas para el sistema web.
# Para publicar reglas limpias:
# python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web

import json

from ..entrenamiento.modelo import RUTA_REGLAS_APRENDIDAS, RUTA_REGLAS_SISTEMA_DIFUSO


def _leer_json(ruta):
    """Lee un JSON de reglas y convierte antecedentes a tuplas."""
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    for r in contenido:
        r["antecedentes"] = [tuple(ant) for ant in r["antecedentes"]]
    return contenido


def _cargar_reglas():
    """Carga reglas publicadas; si no existen, usa el archivo historico."""
    ruta = RUTA_REGLAS_SISTEMA_DIFUSO
    if not ruta.exists():
        ruta = RUTA_REGLAS_APRENDIDAS
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontraron reglas publicadas en {RUTA_REGLAS_SISTEMA_DIFUSO} "
            f"ni reglas historicas en {RUTA_REGLAS_APRENDIDAS}."
        )
    return _leer_json(ruta)


# REGLAS es lo que consume el resto del sistema (motor.py, service.py).
REGLAS = _cargar_reglas()
