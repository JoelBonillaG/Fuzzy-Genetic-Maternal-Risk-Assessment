"""Comparacion experimental de RIPPER, PRISM y AG Pittsburgh-Michigan.

El experimento usa el dataset completo, sin particiones train/test. Todas las
reglas generadas conservan los seis antecedentes clinicos.

Uso:
    python -m riesgo_materno.herramientas.comparaciones.experimento_reglas
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import RUTA_CSV
from ...entrenamiento.prism import aprender_reglas_prism as aprender_prism
from ...entrenamiento.ripper import aprender_reglas_ripper as aprender_ripper
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from ...optimizacion.pittsburgh_michigan import (
    contar_duplicados,
    ejecutar_ag_pittsburgh_michigan,
)


RUTA_BASE = Path(__file__).resolve().parent
RUTA_RESULTADOS = RUTA_BASE / "resultados"

CLASES = ["low risk", "mid risk", "high risk"]
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

CONFIGURACION_EXPERIMENTO = {
    "id_experimento": "comparacion_reglas_riesgo_materno_dataset_completo",
    "iteraciones": 5,
    "clases": CLASES,
    "estrategia_datos": "dataset_completo_sin_splits",
    "metrica_principal": "accuracy",
    "fitness": {
        "peso_balanced_accuracy": 0.98,
        "penalizacion_duplicados": 0.02,
    },
    "ripper": {
        "k": 2,
        "tolerancia_longitud_descripcion": 64,
    },
    "prism": {
        "modo": "prism_bootstrap",
        "fraccion_bootstrap": 1.0,
        "cobertura_minima_regla": 2,
        "orden_clases": CLASES,
        "maximo_condiciones_por_regla": 6,
        "maximo_reglas_por_clase": 20,
        "eliminar_positivos_cubiertos": False,
    },
    "ag_pittsburgh_michigan": {
        "reglas_por_individuo": 368,
        "tamano_poblacion": 30,
        "cantidad_padres": 15,
        "maximo_generaciones": 180,
        "paciencia": 60,
        "probabilidad_cruce": 0.90,
        "probabilidad_mutacion": 0.12,
        "probabilidad_reemplazo": 0.85,
        "fraccion_reemplazo": 0.10,
        "elitismo": 3,
        "tamano_torneo": 3,
        "ruido_clase_inicial": 0.35,
        "peso_balanced_accuracy": 0.98,
        "penalizacion_duplicados": 0.02,
    },
}

def principal():
    ejecutar_experimento(CONFIGURACION_EXPERIMENTO)


def ejecutar_experimento(config):
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar_json(RUTA_RESULTADOS / "config_experimento.json", config)

    tabla = cargar_dataset(RUTA_CSV)
    resultados = []

    print("Dataset completo")
    print(f"  Instancias evaluadas: {len(tabla)}")
    print("  Estrategia: sin split entrenamiento/prueba")

    for iteracion in range(1, config["iteraciones"] + 1):
        print(f"\nIteracion {iteracion:02d}")
        ruta_iteracion = RUTA_RESULTADOS / f"iteracion_{iteracion:02d}"
        ruta_iteracion.mkdir(parents=True, exist_ok=True)

        resultados.append(ejecutar_ripper(iteracion, tabla, ruta_iteracion, config))
        resultados.append(ejecutar_prism(iteracion, tabla, ruta_iteracion, config))
        resultados.append(ejecutar_ag(iteracion, tabla, ruta_iteracion, config))

    resumen = construir_resumen_final(resultados)
    guardar_csv_seguro(pd.DataFrame(resumen["tabla_resumen"]), RUTA_RESULTADOS / "resumen_final.csv")
    guardar_json_seguro(RUTA_RESULTADOS / "resumen_final.json", resumen)
    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")
    return resumen


def ejecutar_ripper(iteracion, tabla, ruta_iteracion, config):
    inicio = time.perf_counter()
    reglas = aprender_ripper(
        tabla,
        orden_clases=CLASES,
        parametros=traducir_parametros_ripper(config["ripper"]),
    )
    asignar_origen(reglas, "RIPPER")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="RIPPER",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "ripper.json",
        hiperparametros=config["ripper"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
    )
    print(
        f"  RIPPER | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f}"
    )
    return resultado


def ejecutar_prism(iteracion, tabla, ruta_iteracion, config):
    inicio = time.perf_counter()
    reglas = aprender_prism_estocastico(tabla, config["prism"])
    asignar_origen(reglas, "PRISM_ESTOCASTICO")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="PRISM_ESTOCASTICO",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "prism.json",
        hiperparametros=config["prism"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
    )
    print(
        f"  PRISM  | modo=bootstrap | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f}"
    )
    return resultado


def aprender_prism_estocastico(tabla, config):
    """Genera un unico conjunto PRISM desde una muestra bootstrap."""
    if config.get("modo") != "prism_bootstrap":
        return aprender_prism(tabla, config)

    fraccion = float(config["fraccion_bootstrap"])
    cantidad_filas = max(1, int(len(tabla) * fraccion))
    muestra = tabla.sample(n=cantidad_filas, replace=True).reset_index(drop=True)
    return aprender_prism(muestra, config)


def ejecutar_ag(iteracion, tabla, ruta_iteracion, config):
    inicio = time.perf_counter()
    print("  AG-PM  | evolucionando base completa de reglas...")
    mejor, historial = ejecutar_ag_pittsburgh_michigan(
        tabla=tabla,
        membresias=construir_membresias_base(),
        parametros=config["ag_pittsburgh_michigan"],
    )
    reglas = mejor.reglas
    asignar_origen(reglas, "AG_PITTSBURGH_MICHIGAN")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="AG_PITTSBURGH_MICHIGAN",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "genetic_algorithm.json",
        hiperparametros=config["ag_pittsburgh_michigan"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
        extra={
            "historial_ag": resumir_historial_ag(historial),
            "mejor_individuo_ag": {
                "cromosoma": [int(gene) for gene in mejor.cromosoma.tolist()],
                "longitud_cromosoma": int(len(mejor.cromosoma)),
                "genes_por_regla": 7,
                "fitness": mejor.fitness,
                "balanced_accuracy": mejor.balanced_accuracy,
                "duplicados": mejor.duplicados,
                "proporcion_duplicados": mejor.proporcion_duplicados,
            },
        },
    )
    print(
        f"  AG-PM  | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f} | "
        f"duplicados={resultado['resumen_reglas']['reglas_duplicadas']}"
    )
    return resultado


def evaluar_y_guardar(
    iteracion,
    algoritmo,
    reglas,
    tabla,
    ruta,
    hiperparametros,
    tiempo_entrenamiento_ms,
    config_fitness,
    extra=None,
):
    inicio_inferencia = time.perf_counter()
    metricas = evaluar_reglas(reglas, tabla)
    tiempo_inferencia_ms = medir_ms(inicio_inferencia)
    resumen_reglas = resumir_reglas(reglas)
    metricas["fitness"] = calcular_fitness(
        balanced_accuracy=metricas["balanced_accuracy"],
        duplicados=resumen_reglas["reglas_duplicadas"],
        total_reglas=resumen_reglas["total_reglas"],
        config_fitness=config_fitness,
    )

    resultado = {
        "id_experimento": CONFIGURACION_EXPERIMENTO["id_experimento"],
        "iteracion": int(iteracion),
        "algoritmo": algoritmo,
        "datos": {
            "estrategia": "dataset_completo_sin_splits",
            "total_instancias": int(len(tabla)),
        },
        "hiperparametros": hiperparametros,
        "resumen_reglas": resumen_reglas,
        "metricas": metricas,
        "matriz_confusion": metricas.pop("matriz_confusion"),
        "reglas_finales": [regla_a_formato_comun(regla, i) for i, regla in enumerate(reglas, start=1)],
        "tiempos": {
            "entrenamiento_ms": int(tiempo_entrenamiento_ms),
            "inferencia_ms": int(tiempo_inferencia_ms),
            "total_ms": int(tiempo_entrenamiento_ms + tiempo_inferencia_ms),
        },
    }
    if extra:
        resultado.update(extra)
    guardar_json(ruta, resultado)
    return resultado


def evaluar_reglas(reglas, tabla):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(construir_membresias_base(), reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    return construir_metricas_desde_predicciones(
        reales=datos["riesgos"],
        predichos=inferencia["riesgos"],
        puntajes=inferencia["puntajes"],
        sin_activacion=inferencia["sin_activacion"],
    )


def construir_metricas_desde_predicciones(reales, predichos, puntajes, sin_activacion):
    correctas = int(np.sum(reales == predichos))
    total = int(len(reales))
    accuracy = float(accuracy_score(reales, predichos))
    matriz = confusion_matrix(reales, predichos, labels=CLASES).tolist()
    return {
        "accuracy": accuracy,
        "error_clasificacion": float(1.0 - accuracy),
        "errores": int(total - correctas),
        "aciertos": correctas,
        "total": total,
        "balanced_accuracy": float(balanced_accuracy_score(reales, predichos)),
        "desviacion_estandar_puntajes": float(np.std(puntajes, ddof=1)),
        "cobertura": float(1.0 - np.mean(sin_activacion)),
        "instancias_sin_activacion": int(np.sum(sin_activacion)),
        "matriz_confusion": {
            "etiquetas": CLASES,
            "matriz": matriz,
        },
    }


def construir_membresias_base():
    membresias = {}
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
    return membresias


def resumir_reglas(reglas):
    total_reglas = len(reglas)
    reglas_duplicadas = contar_duplicados(reglas)
    por_clase = {clase: 0 for clase in CLASES}
    longitudes = []
    for regla in reglas:
        por_clase[CONSECUENTE_A_CLASE[regla["consecuente"]]] += 1
        longitudes.append(len(regla["antecedentes"]))
    return {
        "total_reglas": int(total_reglas),
        "reglas_duplicadas": int(reglas_duplicadas),
        "proporcion_duplicados": float(reglas_duplicadas / total_reglas) if total_reglas else 0.0,
        "antecedentes_por_regla": {
            "minimo": int(min(longitudes)) if longitudes else 0,
            "promedio": float(np.mean(longitudes)) if longitudes else 0.0,
            "maximo": int(max(longitudes)) if longitudes else 0,
        },
        "reglas_por_clase": por_clase,
    }


def calcular_fitness(balanced_accuracy, duplicados, total_reglas, config_fitness):
    proporcion_duplicados = duplicados / total_reglas if total_reglas else 0.0
    return float(
        config_fitness["peso_balanced_accuracy"] * balanced_accuracy
        - config_fitness["penalizacion_duplicados"] * proporcion_duplicados
    )


def traducir_parametros_ripper(parametros):
    return {
        "k": parametros["k"],
        "dl_allowance": parametros["tolerancia_longitud_descripcion"],
    }


def construir_resumen_final(resultados):
    filas = []
    for algoritmo in sorted({r["algoritmo"] for r in resultados}):
        grupo = [r for r in resultados if r["algoritmo"] == algoritmo]
        accuracies = [r["metricas"]["accuracy"] for r in grupo]
        errores = [r["metricas"]["error_clasificacion"] for r in grupo]
        fitness = [r["metricas"]["fitness"] for r in grupo]
        ba = [r["metricas"]["balanced_accuracy"] for r in grupo]
        filas.append(
            {
                "algoritmo": algoritmo,
                "accuracy_promedio": promedio(accuracies),
                "accuracy_desviacion_estandar": desviacion(accuracies),
                "error_promedio": promedio(errores),
                "error_desviacion_estandar": desviacion(errores),
                "balanced_accuracy_promedio": promedio(ba),
                "balanced_accuracy_desviacion_estandar": desviacion(ba),
                "fitness_promedio": promedio(fitness),
                "fitness_desviacion_estandar": desviacion(fitness),
                "reglas_promedio": promedio([r["resumen_reglas"]["total_reglas"] for r in grupo]),
                "duplicados_promedio": promedio([r["resumen_reglas"]["reglas_duplicadas"] for r in grupo]),
                "tiempo_total_ms_promedio": promedio([r["tiempos"]["total_ms"] for r in grupo]),
            }
        )
    return {
        "tabla_resumen": filas,
        "mejores_iteraciones": mejores_iteraciones(resultados, max),
        "peores_iteraciones": mejores_iteraciones(resultados, min),
    }


def mejores_iteraciones(resultados, funcion):
    salida = {}
    for algoritmo in sorted({r["algoritmo"] for r in resultados}):
        grupo = [r for r in resultados if r["algoritmo"] == algoritmo]
        elegido = funcion(grupo, key=lambda r: r["metricas"]["accuracy"])
        salida[algoritmo] = {
            "iteracion": elegido["iteracion"],
            "accuracy": elegido["metricas"]["accuracy"],
            "error_clasificacion": elegido["metricas"]["error_clasificacion"],
            "fitness": elegido["metricas"]["fitness"],
        }
    return salida


def promedio(valores):
    return float(np.mean(valores)) if valores else 0.0


def desviacion(valores):
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0


def resumir_historial_ag(historial):
    filas = historial.to_dict(orient="records") if isinstance(historial, pd.DataFrame) else historial
    if not filas:
        return {
            "mejor_generacion": 0,
            "generacion_final": 0,
            "historial": [],
        }
    mejor = max(filas, key=lambda h: h["mejor_fitness"])
    return {
        "mejor_generacion": int(mejor["generacion"]),
        "generacion_final": int(filas[-1]["generacion"]),
        "fitness_inicial": float(filas[0]["mejor_fitness"]),
        "fitness_final": float(filas[-1]["mejor_fitness"]),
        "historial": filas,
    }


def asignar_origen(reglas, origen):
    for regla in reglas:
        regla["source"] = origen


def regla_a_formato_comun(regla, posicion):
    return {
        "id": f"R{posicion:03d}",
        "antecedentes": [
            {"variable": variable, "etiqueta_linguistica": termino}
            for variable, termino in regla["antecedentes"]
        ],
        "consecuente": CONSECUENTE_A_CLASE[regla["consecuente"]],
        "origen": regla.get("source"),
    }


def guardar_json(ruta, contenido):
    ruta.write_text(
        json.dumps(serializar_resultado(contenido), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def guardar_json_seguro(ruta, contenido):
    try:
        guardar_json(ruta, contenido)
    except PermissionError:
        alternativa = ruta_con_timestamp(ruta)
        guardar_json(alternativa, contenido)
        print(f"  Aviso | {ruta.name} esta bloqueado; se guardo {alternativa.name}")


def guardar_csv_seguro(dataframe, ruta):
    try:
        dataframe.to_csv(ruta, index=False)
    except PermissionError:
        alternativa = ruta_con_timestamp(ruta)
        dataframe.to_csv(alternativa, index=False)
        print(f"  Aviso | {ruta.name} esta bloqueado; se guardo {alternativa.name}")


def ruta_con_timestamp(ruta):
    marca = time.strftime("%Y%m%d_%H%M%S")
    return ruta.with_name(f"{ruta.stem}_{marca}{ruta.suffix}")


def serializar_resultado(valor):
    if isinstance(valor, dict):
        return {k: serializar_resultado(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [serializar_resultado(v) for v in valor]
    if isinstance(valor, tuple):
        return [serializar_resultado(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def medir_ms(inicio):
    return int((time.perf_counter() - inicio) * 1000)


if __name__ == "__main__":
    principal()
