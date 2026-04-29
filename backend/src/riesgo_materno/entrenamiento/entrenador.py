from functools import lru_cache
import json
from pathlib import Path
import threading

import numpy as np

from ..optimizacion.algoritmo_genetico import ejecutar_algoritmo_genetico, evaluar_individuo
from ..optimizacion.cromosoma import CROMOSOMA_BASE, MEMBRESIAS_BASE, decodificar_cromosoma, reparar_cromosoma
from .datos import cargar_datos, convertir_split_a_diccionario, dividir_datos_estratificados
from .evaluacion import crear_tabla_comparativa_prueba, evaluar_membresias_en_splits
from .modelo import RUTA_CSV, RUTA_MODELO_OPTIMIZADO, PARAMETROS_AG

_lock_entrenamiento = threading.Lock()


def _primer_valor(datos: dict, *claves: str, default=0.0):
    for clave in claves:
        if clave in datos:
            return datos[clave]
    return default


def _normalizar_fila_historial(fila: dict) -> dict:
    return {
        "generacion": int(_primer_valor(fila, "generacion", default=0)),
        "fitness": float(_primer_valor(fila, "fitness", "mejor_fitness", default=0.0)),
        "fitness_promedio": float(_primer_valor(fila, "fitness_promedio", default=0.0)),
        "macro_f1": float(_primer_valor(fila, "macro_f1", "macro_f1_validacion", default=0.0)),
        "recall_alto": float(_primer_valor(fila, "recall_alto", "recall_alto_validacion", default=0.0)),
    }


def obtener_resultado_entrenamiento(forzar_reentrenamiento=False, parametros=None):
    """Devuelve el resultado de entrenamiento usando cache. Si forzar_reentrenamiento=True limpia el cache y reentrena."""
    with _lock_entrenamiento:
        if forzar_reentrenamiento:
            entrenar_y_guardar.cache_clear()
        if parametros:
            return entrenar_y_guardar(**parametros)
        return entrenar_y_guardar()


def obtener_membresias_optimizadas():
    """Carga las membresias optimizadas desde disco si existen, o entrena desde cero."""
    resultado = cargar_modelo_optimizado()
    if resultado is not None:
        return resultado["membresias_optimizadas"], "modelo persistido en disco"

    resultado = entrenar_y_guardar()
    return resultado["membresias_optimizadas"], "modelo entrenado y guardado desde el CSV"


@lru_cache(maxsize=1)
def entrenar_y_guardar(
    tamano_poblacion=PARAMETROS_AG["tamano_poblacion"],
    cantidad_hijos=PARAMETROS_AG["cantidad_hijos"],
    maximo_generaciones=PARAMETROS_AG["maximo_generaciones"],
    probabilidad_cruce=PARAMETROS_AG["probabilidad_cruce"],
    probabilidad_mutacion=PARAMETROS_AG["probabilidad_mutacion"],
):
    """Carga datos, optimiza el AG con entrenamiento, elige por validacion y guarda el mejor cromosoma."""
    parametros_override = {
        "tamano_poblacion": tamano_poblacion,
        "cantidad_hijos": cantidad_hijos,
        "maximo_generaciones": maximo_generaciones,
        "probabilidad_cruce": probabilidad_cruce,
        "probabilidad_mutacion": probabilidad_mutacion,
    }

    datos = cargar_datos(RUTA_CSV)
    splits = dividir_datos_estratificados(datos)
    #print("Splits creados:", splits.keys())
    #print("Columnas de entrenamiento:", splits["entrenamiento"].columns)

    datos_por_split = {}
    for nombre, tabla in splits.items():
        datos_convertidos = convertir_split_a_diccionario(tabla)
        datos_por_split[nombre] = datos_convertidos

    mejor_individuo, historial = ejecutar_algoritmo_genetico(
        datos_por_split["entrenamiento"],
        datos_por_split["validacion"],
        parametros_override=parametros_override,
    )
    
    membresias_optimizadas = decodificar_cromosoma(mejor_individuo.cromosoma)

    resultados_base = evaluar_membresias_en_splits(MEMBRESIAS_BASE, datos_por_split)
    resultados_optimizados = evaluar_membresias_en_splits(membresias_optimizadas, datos_por_split)

    fitness_base = evaluar_individuo(CROMOSOMA_BASE, datos_por_split["validacion"]).fitness
    fitness_opt = mejor_individuo.fitness

    tabla_comparativa = crear_tabla_comparativa_prueba(
        resultados_base["validacion"],
        resultados_optimizados["validacion"],
        fitness_base=fitness_base,
        fitness_opt=fitness_opt,
    )

    resultado = {
        "splits": splits,
        "datos_por_split": datos_por_split,
        "membresias_base": MEMBRESIAS_BASE,
        "membresias_optimizadas": membresias_optimizadas,
        "resultados_base": resultados_base,
        "resultados_optimizados": resultados_optimizados,
        "tabla_comparativa": tabla_comparativa,
        "mejor_individuo": mejor_individuo,
        "historial": historial,
    }
    guardar_modelo_optimizado(resultado)
    return resultado


def entrenar_con_progreso(parametros: dict, progress_callback=None):
    """Igual que entrenar_y_guardar pero acepta un callback de progreso por generacion.
    No usa lru_cache para poder recibir el callback. Al terminar limpia el cache."""
    parametros_override = {
        "tamano_poblacion": parametros.get("tamano_poblacion", PARAMETROS_AG["tamano_poblacion"]),
        "cantidad_hijos": parametros.get("cantidad_hijos", PARAMETROS_AG["cantidad_hijos"]),
        "maximo_generaciones": parametros.get("maximo_generaciones", PARAMETROS_AG["maximo_generaciones"]),
        "probabilidad_cruce": parametros.get("probabilidad_cruce", PARAMETROS_AG["probabilidad_cruce"]),
        "probabilidad_mutacion": parametros.get("probabilidad_mutacion", PARAMETROS_AG["probabilidad_mutacion"]),
    }
    datos = cargar_datos(RUTA_CSV)
    splits = dividir_datos_estratificados(datos)
    datos_por_split = {
        nombre: convertir_split_a_diccionario(tabla)
        for nombre, tabla in splits.items()
    }
    mejor_individuo, historial = ejecutar_algoritmo_genetico(
        datos_por_split["entrenamiento"],
        datos_por_split["validacion"],
        parametros_override=parametros_override,
        progress_callback=progress_callback,
    )
    membresias_optimizadas = decodificar_cromosoma(mejor_individuo.cromosoma)

    resultados_base = evaluar_membresias_en_splits(MEMBRESIAS_BASE, datos_por_split)
    resultados_optimizados = evaluar_membresias_en_splits(membresias_optimizadas, datos_por_split)

    fitness_base = evaluar_individuo(CROMOSOMA_BASE, datos_por_split["validacion"]).fitness
    fitness_opt = mejor_individuo.fitness

    tabla_comparativa = crear_tabla_comparativa_prueba(
        resultados_base["validacion"],
        resultados_optimizados["validacion"],
        fitness_base=fitness_base,
        fitness_opt=fitness_opt,
    )

    resultado = {
        "splits": splits,
        "datos_por_split": datos_por_split,
        "membresias_base": MEMBRESIAS_BASE,
        "membresias_optimizadas": membresias_optimizadas,
        "resultados_base": resultados_base,
        "resultados_optimizados": resultados_optimizados,
        "tabla_comparativa": tabla_comparativa,
        "mejor_individuo": mejor_individuo,
        "historial": historial,
    }
    guardar_modelo_optimizado(resultado)
    entrenar_y_guardar.cache_clear()
    return resultado


def cargar_modelo_optimizado():
    """Lee modelo_optimizado.json y devuelve las membresias y metricas guardadas, o None si no existe."""
    ruta_modelo = Path(RUTA_MODELO_OPTIMIZADO)
    if not ruta_modelo.exists():
        return None

    datos_modelo = json.loads(ruta_modelo.read_text(encoding="utf-8"))
    cromosoma = np.asarray(datos_modelo["mejor_cromosoma"], dtype=float)
    cromosoma_reparado = reparar_cromosoma(cromosoma)
    if np.isnan(cromosoma_reparado).any():
        return None

    return {
        "membresias_optimizadas": decodificar_cromosoma(cromosoma_reparado),
        "mejor_cromosoma": cromosoma_reparado,
        "fuente_modelo": "disco",
        "ruta_modelo": str(ruta_modelo),
        "fitness": datos_modelo.get("fitness", 0.0),
        "macro_f1": _primer_valor(datos_modelo, "macro_f1", "macro_f1_validacion", default=0.0),
        "recall_alto": _primer_valor(datos_modelo, "recall_alto", "recall_alto_validacion", default=0.0),
        "generaciones": datos_modelo.get("generaciones", 0),
        "historial": [
            _normalizar_fila_historial(fila)
            for fila in datos_modelo.get("historial", [])
        ],
        "tabla_comparativa": datos_modelo.get("tabla_comparativa", []),
    }


def guardar_modelo_optimizado(resultado):
    """Serializa el mejor cromosoma, fitness, historial y tabla comparativa en modelo_optimizado.json."""
    mejor_individuo = resultado["mejor_individuo"]
    historial = resultado["historial"]
    tabla_comparativa = resultado["tabla_comparativa"]

    def _serializable(v):
        """Convierte valores numpy a tipos nativos de Python."""
        if hasattr(v, "item"):
            return v.item()
        return v

    historial_lista = [
        {k: _serializable(v) for k, v in fila.items()}
        for fila in historial.to_dict(orient="records")
    ]

    comparativa_lista = [
        {k: _serializable(v) for k, v in fila.items()}
        for fila in tabla_comparativa.to_dict(orient="records")
    ]

    contenido = {
        "ruta_csv": str(RUTA_CSV),
        "mejor_cromosoma": [float(valor) for valor in mejor_individuo.cromosoma.tolist()],
        "fitness": float(mejor_individuo.fitness),
        "macro_f1": float(mejor_individuo.macro_f1_validacion),
        "recall_alto": float(mejor_individuo.recall_alto_validacion),
        "generaciones": int(len(historial) - 1),
        "historial": [_normalizar_fila_historial(fila) for fila in historial_lista],
        "tabla_comparativa": comparativa_lista,
    }
    Path(RUTA_MODELO_OPTIMIZADO).write_text(
        json.dumps(contenido, indent=2),
        encoding="utf-8",
    )
