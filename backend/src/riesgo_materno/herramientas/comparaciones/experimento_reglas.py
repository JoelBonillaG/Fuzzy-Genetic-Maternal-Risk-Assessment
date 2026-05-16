"""Comparacion experimental de RIPPER, PRISM y AG Michigan binario.

El experimento usa el dataset completo, sin particiones train/test. En esta
rama las reglas conservan seis antecedentes clinicos.

Uso:
    python -m riesgo_materno.herramientas.comparaciones.experimento_reglas
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import RUTA_CSV
from ...entrenamiento.prism import aprender_reglas_prism as aprender_prism
from ...entrenamiento.ripper import aprender_reglas_ripper as aprender_ripper
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from ...optimizacion.michigan_binario import (
    BITS_POR_GEN,
    BITS_POR_REGLA,
    BITS_POR_CONSECUENTE,
    contar_duplicados,
    ejecutar_ag_michigan_binario,
)


RUTA_BASE = Path(__file__).resolve().parent
RUTA_RESULTADOS = RUTA_BASE / "resultados"

CLASES = ["low risk", "mid risk", "high risk"]
CLASE_SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}
CONFIGURACION_EXPERIMENTO = {
    "id_experimento": "prueba_michigan_binario_recall_precision_rapida",
    "iteraciones": 20,
    "clases": CLASES,
    "estrategia_datos": "dataset_completo_sin_splits",
    "metrica_principal": "balanced_accuracy",

    "ripper": {
        "k": 2,
        "tolerancia_longitud_descripcion": 64,
    },

    "prism": {
        "modo": "prism_bootstrap",
        "fraccion_bootstrap": 0.75,
        "cobertura_minima_regla": 2,
        "orden_clases": CLASES,
        "maximo_condiciones_por_regla": 6,
        "maximo_reglas_por_clase": 40,
        "eliminar_positivos_cubiertos": True,
    },

    "ag_michigan_binario": {
        "reglas_por_poblacion": 368,
        "bits_por_gen": 3,
        "cantidad_padres": 120,
        "maximo_generaciones": 1500,
        "paciencia": 1000,
        "probabilidad_cruce": 0.90,
        "probabilidad_mutacion": 0.05,
        "elitismo": 12,
        "tamano_torneo": 5,
        "balancear_consecuentes_por_clase": True,
        "usar_fitness_compuesto": True,
        "peso_calidad_local": 0.45,
        "peso_aporte_clase": 0.35,
        "peso_confusion_otras_clases": 0.15,
        "peso_penalizacion_duplicado": 0.005,
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
    reglas = aprender_ripper(
        tabla,
        orden_clases=CLASES,
        parametros=traducir_parametros_ripper(config["ripper"]),
    )
    asignar_origen(reglas, "RIPPER")
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="RIPPER",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "ripper.json",
        hiperparametros=config["ripper"],
        config=config,
    )
    print(
        f"  RIPPER | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"ba={resultado['metricas']['balanced_accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f} | "
        f"error_ba={resultado['metricas']['error_balanceado']:.4f}"
    )
    return resultado


def ejecutar_prism(iteracion, tabla, ruta_iteracion, config):
    reglas = aprender_prism_estocastico(tabla, config["prism"])
    asignar_origen(reglas, "PRISM_ESTOCASTICO")
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="PRISM_ESTOCASTICO",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "prism.json",
        hiperparametros=config["prism"],
        config=config,
    )
    print(
        f"  PRISM  | modo=bootstrap | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"ba={resultado['metricas']['balanced_accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f} | "
        f"error_ba={resultado['metricas']['error_balanceado']:.4f}"
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
    print("  AG-MB  | evolucionando poblacion de reglas binarias...")
    mejor, historial = ejecutar_ag_michigan_binario(
        tabla=tabla,
        membresias=construir_membresias_base(),
        parametros=config["ag_michigan_binario"],
    )
    reglas = mejor.reglas
    asignar_origen(reglas, "AG_MICHIGAN_BINARIO")
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="AG_MICHIGAN_BINARIO",
        reglas=reglas,
        tabla=tabla,
        ruta=ruta_iteracion / "genetic_algorithm.json",
        hiperparametros=config["ag_michigan_binario"],
        config=config,
        extra={
            "historial_ag": resumir_historial_ag(historial),
            "mejor_poblacion_ag": {
                "bits_por_gen": BITS_POR_GEN,
                "bits_por_consecuente": BITS_POR_CONSECUENTE,
                "bits_por_regla": BITS_POR_REGLA,
                "intentos_invalidos_descartados_generacion": mejor.intentos_invalidos_descartados_generacion,
                "cromosomas": [
                    {
                        "id": f"R{indice:03d}",
                        "bits": "".join(str(int(bit)) for bit in individuo.cromosoma.tolist()),
                        "consecuente": individuo.clase,
                    }
                    for indice, individuo in enumerate(mejor.individuos, start=1)
                ],
            },
        },
    )
    print(
        f"  AG-MB  | reglas={len(reglas)} | "
        f"accuracy={resultado['metricas']['accuracy']:.4f} | "
        f"ba={resultado['metricas']['balanced_accuracy']:.4f} | "
        f"error={resultado['metricas']['error_clasificacion']:.4f} | "
        f"error_ba={resultado['metricas']['error_balanceado']:.4f} | "
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
    config,
    extra=None,
):
    metricas = evaluar_reglas(reglas, tabla)
    resumen_reglas = resumir_reglas(reglas)
    resultado = {
        "id_experimento": config["id_experimento"],
        "iteracion": int(iteracion),
        "algoritmo": algoritmo,
        "datos": {
            "estrategia": config["estrategia_datos"],
            "total_instancias": int(len(tabla)),
        },
        "hiperparametros": hiperparametros,
        "resumen_reglas": resumen_reglas,
        "metricas": metricas,
        "matriz_confusion": metricas.pop("matriz_confusion"),
        "reglas_finales": [regla_a_formato_comun(regla, i) for i, regla in enumerate(reglas, start=1)],
    }
    if extra:
        resultado.update(extra)
    guardar_json(ruta, resultado)
    return resultado


def evaluar_reglas(reglas, tabla):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(
        construir_membresias_base(),
        reglas=reglas,
        permitir_neutro=False,
    )
    inferencia = sistema.inferir_lote(datos["entradas"])
    return construir_metricas_desde_predicciones(
        reales=datos["riesgos"],
        predichos=inferencia["riesgos"],
        sin_activacion=inferencia["sin_activacion"],
    )


def construir_metricas_desde_predicciones(reales, predichos, sin_activacion=None):
    predichos = np.asarray(predichos, dtype=object).copy()
    if sin_activacion is not None:
        predichos[np.asarray(sin_activacion, dtype=bool)] = CLASE_SIN_ACTIVACION

    correctas = int(np.sum(reales == predichos))
    total = int(len(reales))
    etiquetas_matriz = CLASES + [CLASE_SIN_ACTIVACION]
    accuracy = float(accuracy_score(reales, predichos))
    matriz = confusion_matrix(reales, predichos, labels=etiquetas_matriz).tolist()
    balanced_accuracy = balanced_accuracy_con_clases_reales(reales, predichos)
    return {
        "accuracy": accuracy,
        "error_clasificacion": float(1.0 - accuracy),
        "errores": int(total - correctas),
        "aciertos": correctas,
        "total": total,
        "sin_activacion": int(np.sum(predichos == CLASE_SIN_ACTIVACION)),
        "balanced_accuracy": balanced_accuracy,
        "error_balanceado": float(1.0 - balanced_accuracy),
        "matriz_confusion": {
            "etiquetas": etiquetas_matriz,
            "matriz": matriz,
        },
    }


def balanced_accuracy_con_clases_reales(reales, predichos):
    recalls = []
    for clase in CLASES:
        mascara_clase = reales == clase
        total_clase = int(np.sum(mascara_clase))
        if total_clase == 0:
            continue
        verdaderos_positivos = int(np.sum(predichos[mascara_clase] == clase))
        recalls.append(verdaderos_positivos / total_clase)
    return float(np.mean(recalls)) if recalls else 0.0


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
        "reglas_por_clase": por_clase,
    }


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
        errores_balanceados = [r["metricas"]["error_balanceado"] for r in grupo]
        ba = [r["metricas"]["balanced_accuracy"] for r in grupo]
        filas.append(
            {
                "algoritmo": algoritmo,
                "accuracy_promedio": promedio(accuracies),
                "accuracy_desviacion_estandar": desviacion(accuracies),
                "error_promedio": promedio(errores),
                "error_desviacion_estandar": desviacion(errores),
                "error_balanceado_promedio": promedio(errores_balanceados),
                "error_balanceado_desviacion_estandar": desviacion(errores_balanceados),
                "balanced_accuracy_promedio": promedio(ba),
                "balanced_accuracy_desviacion_estandar": desviacion(ba),
            }
        )
    return {
        "tabla_resumen": filas,
    }


def promedio(valores):
    return float(np.mean(valores)) if valores else 0.0


def desviacion(valores):
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0


def resumir_historial_ag(historial):
    filas = historial.to_dict(orient="records") if isinstance(historial, pd.DataFrame) else historial
    return {
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


if __name__ == "__main__":
    principal()
