"""Orquestador del entrenamiento Pittsburgh: corre el AG y persiste la seleccion de reglas."""

import json
from pathlib import Path

import numpy as np

from ..logica_difusa.reglas import REGLAS
from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from ..optimizacion.algoritmo_genetico import (
    ResultadoBase,
    ejecutar_ag_pittsburgh,
    evaluar_base_reglas,
)
from ..optimizacion.cromosoma import (
    CANTIDAD_REGLAS_CANDIDATAS,
    indices_reglas_activas,
    numeros_reglas_activas,
)
from .datos import (
    cargar_dataset,
    convertir_split_a_diccionario,
    dividir_entrenamiento_prueba,
    resumir_splits,
)
from .modelo import RUTA_CSV, RUTA_SELECCION_REGLAS, PARAMETROS_AG


def entrenar_seleccion_reglas(parametros_override=None, progress_callback=None, semilla=None):
    """Carga el CSV, divide 70/30, corre el AG sobre entrenamiento, evalua en prueba y guarda la seleccion.

    Devuelve un dict con todos los artefactos del entrenamiento.
    """
    membresias = construir_membresias_base()

    tabla = cargar_dataset(RUTA_CSV)
    splits = dividir_entrenamiento_prueba(tabla, semilla=semilla)

    datos_por_split = {}
    for nombre, tabla_split in splits.items():
        datos_por_split[nombre] = convertir_split_a_diccionario(tabla_split)

    mejor, historial = ejecutar_ag_pittsburgh(
        datos_entrenamiento=datos_por_split["entrenamiento"],
        membresias=membresias,
        parametros_override=parametros_override,
        progress_callback=progress_callback,
    )

    resultado_prueba = evaluar_base_reglas(
        mejor.cromosoma, datos_por_split["prueba"], membresias
    )

    resultado = {
        "mejor": mejor,
        "resultado_prueba": resultado_prueba,
        "historial": historial,
        "splits": splits,
        "resumen_splits": resumir_splits(splits),
        "membresias": membresias,
    }
    guardar_seleccion_reglas(resultado)
    return resultado


def construir_membresias_base():
    """Membresias trapezoidales tomadas tal cual desde ESPECIFICACIONES_VARIABLES (no se optimizan)."""
    membresias = {}
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        categorias = {}
        for categoria, puntos in especificacion["categorias"].items():
            categorias[categoria] = np.asarray(puntos, dtype=float)
        membresias[variable] = categorias
    return membresias


def guardar_seleccion_reglas(resultado):
    """Serializa el cromosoma ganador, metricas en prueba e historial en modelo_optimizado_reglas.json."""
    mejor = resultado["mejor"]
    resultado_prueba = resultado["resultado_prueba"]
    historial = resultado["historial"]

    total_prueba = len(resultado["splits"]["prueba"])
    total_entrenamiento = len(resultado["splits"]["entrenamiento"])

    cromosoma = np.asarray(mejor.cromosoma, dtype=int).copy()
    cantidad_reglas = cantidad_reglas_activas(cromosoma)
    if cantidad_reglas != int(mejor.cantidad_reglas):
        raise ValueError(
            "Inconsistencia al guardar: el cromosoma no coincide con "
            "la cantidad de reglas del resultado evaluado."
        )

    cromosoma_serializable = []
    for bit in cromosoma.tolist():
        cromosoma_serializable.append(int(bit))

    historial_serializable = []
    for fila in historial.to_dict(orient="records"):
        fila_limpia = {}
        for clave, valor in fila.items():
            fila_limpia[clave] = _a_python(valor)
        historial_serializable.append(fila_limpia)

    contenido = {
        "ruta_csv": str(RUTA_CSV),
        "cantidad_reglas_candidatas": CANTIDAD_REGLAS_CANDIDATAS,
        "cromosoma": cromosoma_serializable,
        "indices_reglas_activas": indices_reglas_activas(cromosoma),
        "numeros_reglas_activas": numeros_reglas_activas(cromosoma),
        "cantidad_reglas": int(cantidad_reglas),
        "fitness": float(mejor.fitness),
        "balanced_accuracy_entrenamiento": float(mejor.balanced_accuracy),
        "compacidad_entrenamiento": float(mejor.compacidad),
        "total_entrenamiento": total_entrenamiento,
        "metricas_prueba": {
            "balanced_accuracy": float(resultado_prueba.balanced_accuracy),
            "compacidad": float(resultado_prueba.compacidad),
            "cantidad_reglas": int(resultado_prueba.cantidad_reglas),
            "fitness": float(resultado_prueba.fitness),
        },
        "generaciones_ejecutadas": int(len(historial) - 1),
        "historial": historial_serializable,
    }
    Path(RUTA_SELECCION_REGLAS).write_text(
        json.dumps(contenido, indent=2),
        encoding="utf-8",
    )


def _a_python(valor):
    """Convierte numpy scalar a tipo nativo de Python para serializar a JSON."""
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def cargar_seleccion_reglas():
    """Lee el JSON de reglas seleccionadas y devuelve la base activa.

    Devuelve None si el archivo no existe.
    """
    ruta = Path(RUTA_SELECCION_REGLAS)
    if not ruta.exists():
        return None

    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    cromosoma = np.asarray(contenido["cromosoma"], dtype=int)

    if len(cromosoma) != CANTIDAD_REGLAS_CANDIDATAS:
        raise ValueError(
            f"El cromosoma guardado tiene {len(cromosoma)} bits pero hay "
            f"{CANTIDAD_REGLAS_CANDIDATAS} reglas candidatas. Reentrene."
        )

    indices_activos = indices_reglas_activas(cromosoma)
    reglas_activas = []
    for i in indices_activos:
        reglas_activas.append(REGLAS[i])

    return {
        "cromosoma": cromosoma,
        "reglas_activas": reglas_activas,
        "numeros_reglas_activas": numeros_reglas_activas(cromosoma),
        "fitness": float(contenido.get("fitness", 0.0)),
        "aciertos_entrenamiento": int(contenido.get("aciertos_entrenamiento", 0)),
        "cantidad_reglas": int(contenido.get("cantidad_reglas", len(reglas_activas))),
        "metricas_prueba": contenido.get("metricas_prueba", {}),
        "historial": contenido.get("historial", []),
        "ruta_modelo": str(ruta),
    }
