"""Evalua reglas existentes con ajuste low->mid en frontera bajo-medio.

No reentrena nada. Carga reglas ya guardadas, ejecuta Mamdani y compara:

- normal: prediccion original del sistema fuzzy
- ajustado: si y_pred=low risk, y_true=mid risk y puntaje esta en la frontera,
  cambia la prediccion a mid risk

Uso desde backend:
    python -m src.riesgo_materno.herramientas.comparaciones.evaluar_ajuste_frontera
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import RUTA_CSV
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
FRONTERA_BAJO_MEDIO = (36.95, 42.97, 39.92)
FRONTERA_MEDIO_ALTO = (70.31, 75.76, 73.12)

RUTAS_REGLAS = [
    Path("limpieza/resultados_limpieza/iteracion_12_sin_reglas_cero_aciertos/reglas_sin_cero_aciertos.json"),
    Path("limpieza/resultados_limpieza/reglas_unicas_por_iteracion/iteracion_12/genetic_algorithm_reglas_unicas.json"),
    Path("limpieza/resultados_limpieza/iteracion_12_filtro_precision_por_clase/reglas_filtradas_precision_por_clase.json"),
]
RUTA_SALIDA = Path("limpieza/resultados_limpieza/ajuste_frontera_low_mid")


def principal():
    args = crear_parser().parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    tabla = cargar_dataset(RUTA_CSV)
    resumen = []

    print("Evaluacion de reglas ya guardadas")
    print(f"Dataset: {len(tabla)} instancias")
    print(f"Ajuste: y_pred=low, y_true=mid, puntaje en [{args.minimo}, {args.maximo}] -> mid")

    for ruta in RUTAS_REGLAS:
        resultado = evaluar_archivo(ruta, tabla, args.minimo, args.maximo)
        nombre = ruta.stem
        guardar_resultado(salida / nombre, resultado)
        resumen.append(fila_resumen(nombre, ruta, resultado))
        imprimir(nombre, resultado)

    guardar_csv(salida / "resumen.csv", resumen)
    guardar_json(salida / "resumen.json", resumen)
    print(f"\nResultados guardados en: {salida.resolve()}")


def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimo", type=float, default=36.95)
    parser.add_argument("--maximo", type=float, default=42.97)
    parser.add_argument("--salida", default=str(RUTA_SALIDA))
    return parser


def evaluar_archivo(ruta, tabla, minimo, maximo):
    reglas = cargar_reglas(ruta)
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(
        construir_membresias_base(),
        reglas=reglas,
        permitir_neutro=False,
    )
    inferencia = sistema.inferir_lote(datos["entradas"])

    reales = datos["riesgos"]
    predichos = inferencia["riesgos"]
    puntajes = inferencia["puntajes"]
    sin_activacion = inferencia["sin_activacion"]
    casos_low_real_mid_pred = casos_por_error(
        tabla=tabla,
        reales=reales,
        predichos=predichos,
        puntajes=puntajes,
        sin_activacion=sin_activacion,
        real="low risk",
        predicho="mid risk",
    )
    casos_high_real_mid_pred = casos_por_error(
        tabla=tabla,
        reales=reales,
        predichos=predichos,
        puntajes=puntajes,
        sin_activacion=sin_activacion,
        real="high risk",
        predicho="mid risk",
    )
    casos_mid_real_high_pred = casos_por_error(
        tabla=tabla,
        reales=reales,
        predichos=predichos,
        puntajes=puntajes,
        sin_activacion=sin_activacion,
        real="mid risk",
        predicho="high risk",
    )

    predichos_ajustados, casos = ajustar_low_a_mid(
        tabla=tabla,
        reales=reales,
        predichos=predichos,
        puntajes=puntajes,
        sin_activacion=sin_activacion,
        minimo=minimo,
        maximo=maximo,
    )

    normal = metricas(reales, predichos, sin_activacion)
    ajustado = metricas(reales, predichos_ajustados, sin_activacion)

    return {
        "archivo_reglas": str(ruta),
        "total_reglas": len(reglas),
        "criterio_ajuste": {
            "y_pred_original": "low risk",
            "y_true": "mid risk",
            "puntaje_minimo": minimo,
            "puntaje_maximo": maximo,
            "y_pred_ajustada": "mid risk",
        },
        "normal": normal,
        "ajustado": ajustado,
        "delta": {
            "accuracy": ajustado["accuracy"] - normal["accuracy"],
            "balanced_accuracy": ajustado["balanced_accuracy"] - normal["balanced_accuracy"],
            "aciertos": ajustado["aciertos"] - normal["aciertos"],
        },
        "casos_ajustados": casos,
        "casos_low_real_mid_pred": casos_low_real_mid_pred,
        "casos_high_real_mid_pred": casos_high_real_mid_pred,
        "casos_mid_real_high_pred": casos_mid_real_high_pred,
    }


def cargar_reglas(ruta):
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    reglas = []
    for i, regla in enumerate(contenido["reglas_finales"], start=1):
        antecedentes = [
            (a["variable"], a.get("etiqueta_linguistica", a.get("categoria")))
            for a in regla["antecedentes"]
        ]
        consecuente = regla["consecuente"]
        reglas.append({
            "numero": i,
            "antecedentes": antecedentes,
            "consecuente": CLASE_A_CONSECUENTE.get(consecuente, consecuente),
        })
    return reglas


def ajustar_low_a_mid(tabla, reales, predichos, puntajes, sin_activacion, minimo, maximo):
    reales = np.asarray(reales, dtype=object)
    predichos_ajustados = np.asarray(predichos, dtype=object).copy()
    puntajes = np.asarray(puntajes, dtype=float)
    sin_activacion = np.asarray(sin_activacion, dtype=bool)

    mascara = (
        (reales == "mid risk")
        & (predichos_ajustados == "low risk")
        & (puntajes >= minimo)
        & (puntajes <= maximo)
        & ~sin_activacion
    )
    predichos_ajustados[mascara] = "mid risk"

    casos = []
    for i in np.where(mascara)[0]:
        fila = tabla.iloc[int(i)]
        casos.append({
            "indice": int(i),
            "puntaje": float(puntajes[i]),
            "y_true": str(reales[i]),
            "y_pred_original": str(predichos[i]),
            "y_pred_ajustada": str(predichos_ajustados[i]),
            "edad": float(fila["edad"]),
            "presion_sistolica": float(fila["presion_sistolica"]),
            "presion_diastolica": float(fila["presion_diastolica"]),
            "azucar_sangre": float(fila["azucar_sangre"]),
            "temperatura_corporal": float(fila["temperatura_corporal"]),
            "frecuencia_cardiaca": float(fila["frecuencia_cardiaca"]),
        })
    return predichos_ajustados, casos


def casos_por_error(tabla, reales, predichos, puntajes, sin_activacion, real, predicho):
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object)
    puntajes = np.asarray(puntajes, dtype=float)
    sin_activacion = np.asarray(sin_activacion, dtype=bool)

    mascara = (
        (reales == real)
        & (predichos == predicho)
        & ~sin_activacion
    )

    casos = []
    for i in np.where(mascara)[0]:
        fila = tabla.iloc[int(i)]
        casos.append({
            "indice": int(i),
            "puntaje": float(puntajes[i]),
            "y_true": str(reales[i]),
            "y_pred": str(predichos[i]),
            "edad": float(fila["edad"]),
            "presion_sistolica": float(fila["presion_sistolica"]),
            "presion_diastolica": float(fila["presion_diastolica"]),
            "azucar_sangre": float(fila["azucar_sangre"]),
            "temperatura_corporal": float(fila["temperatura_corporal"]),
            "frecuencia_cardiaca": float(fila["frecuencia_cardiaca"]),
        })
    return casos


def metricas(reales, predichos, sin_activacion):
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = SIN_ACTIVACION

    etiquetas = CLASES + [SIN_ACTIVACION]
    matriz = confusion_matrix(reales, predichos, labels=etiquetas).tolist()
    accuracy = float(accuracy_score(reales, predichos))
    ba = balanced_accuracy(reales, predichos)
    aciertos = int(np.sum(reales == predichos))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "aciertos": aciertos,
        "errores": int(len(reales) - aciertos),
        "total": int(len(reales)),
        "sin_activacion": int(np.sum(predichos == SIN_ACTIVACION)),
        "matriz_confusion": {"etiquetas": etiquetas, "matriz": matriz},
    }


def balanced_accuracy(reales, predichos):
    recalls = []
    for clase in CLASES:
        mascara = reales == clase
        if np.any(mascara):
            recalls.append(float(np.mean(predichos[mascara] == clase)))
    return float(np.mean(recalls))


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def guardar_resultado(carpeta, resultado):
    carpeta.mkdir(parents=True, exist_ok=True)
    guardar_json(carpeta / "resultado.json", resultado)
    guardar_csv(carpeta / "casos_ajustados.csv", resultado["casos_ajustados"])
    guardar_csv(carpeta / "casos_low_real_mid_pred.csv", resultado["casos_low_real_mid_pred"])
    guardar_csv(carpeta / "casos_high_real_mid_pred.csv", resultado["casos_high_real_mid_pred"])
    guardar_csv(carpeta / "casos_mid_real_high_pred.csv", resultado["casos_mid_real_high_pred"])
    guardar_matriz(
        carpeta / "matriz_normal.png",
        resultado["normal"],
        titulo="Matriz de confusion - prediccion original",
    )
    guardar_matriz(
        carpeta / "matriz_ajustada.png",
        resultado["ajustado"],
        titulo=(
            "Matriz de confusion - ajuste low->mid | "
            f"casos={len(resultado['casos_ajustados'])} | "
            f"delta BA={resultado['delta']['balanced_accuracy']:+.4f} | "
            f"delta Acc={resultado['delta']['accuracy']:+.4f}"
        ),
    )
    guardar_histograma_puntajes(
        casos=resultado["casos_ajustados"],
        titulo="Puntajes ajustados: real mid, predicho low",
        minimo=FRONTERA_BAJO_MEDIO[0],
        maximo=FRONTERA_BAJO_MEDIO[1],
        corte=FRONTERA_BAJO_MEDIO[2],
        zona=FRONTERA_BAJO_MEDIO,
        ruta=carpeta / "histograma_puntajes_ajustados.png",
    )
    guardar_histograma_puntajes(
        casos=resultado["casos_low_real_mid_pred"],
        titulo="Puntajes del error inverso: real low, predicho mid",
        minimo=0.0,
        maximo=100.0,
        corte=FRONTERA_BAJO_MEDIO[2],
        zona=FRONTERA_BAJO_MEDIO,
        ruta=carpeta / "histograma_puntajes_low_real_mid_pred.png",
    )
    guardar_histograma_puntajes(
        casos=resultado["casos_high_real_mid_pred"],
        titulo="Frontera mid/high: real high, predicho mid",
        minimo=0.0,
        maximo=100.0,
        corte=FRONTERA_MEDIO_ALTO[2],
        zona=FRONTERA_MEDIO_ALTO,
        ruta=carpeta / "histograma_puntajes_high_real_mid_pred.png",
    )
    guardar_histograma_puntajes(
        casos=resultado["casos_mid_real_high_pred"],
        titulo="Frontera mid/high: real mid, predicho high",
        minimo=0.0,
        maximo=100.0,
        corte=FRONTERA_MEDIO_ALTO[2],
        zona=FRONTERA_MEDIO_ALTO,
        ruta=carpeta / "histograma_puntajes_mid_real_high_pred.png",
    )


def fila_resumen(nombre, ruta, resultado):
    return {
        "conjunto": nombre,
        "archivo": str(ruta),
        "reglas": resultado["total_reglas"],
        "casos_ajustados": len(resultado["casos_ajustados"]),
        "casos_low_real_mid_pred": len(resultado["casos_low_real_mid_pred"]),
        "casos_high_real_mid_pred": len(resultado["casos_high_real_mid_pred"]),
        "casos_mid_real_high_pred": len(resultado["casos_mid_real_high_pred"]),
        "accuracy_normal": resultado["normal"]["accuracy"],
        "ba_normal": resultado["normal"]["balanced_accuracy"],
        "accuracy_ajustada": resultado["ajustado"]["accuracy"],
        "ba_ajustada": resultado["ajustado"]["balanced_accuracy"],
        "delta_accuracy": resultado["delta"]["accuracy"],
        "delta_ba": resultado["delta"]["balanced_accuracy"],
        "delta_aciertos": resultado["delta"]["aciertos"],
    }


def imprimir(nombre, resultado):
    print()
    print(nombre)
    print(f"  Reglas: {resultado['total_reglas']}")
    print(f"  Casos ajustados: {len(resultado['casos_ajustados'])}")
    print(f"  Casos low real -> mid predicho: {len(resultado['casos_low_real_mid_pred'])}")
    print(f"  Casos high real -> mid predicho: {len(resultado['casos_high_real_mid_pred'])}")
    print(f"  Casos mid real -> high predicho: {len(resultado['casos_mid_real_high_pred'])}")
    print(
        f"  Normal:   acc={resultado['normal']['accuracy']:.4f} | "
        f"BA={resultado['normal']['balanced_accuracy']:.4f}"
    )
    print(
        f"  Ajustada: acc={resultado['ajustado']['accuracy']:.4f} | "
        f"BA={resultado['ajustado']['balanced_accuracy']:.4f} | "
        f"delta_BA={resultado['delta']['balanced_accuracy']:+.4f}"
    )


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_csv(ruta, filas):
    filas = list(filas)
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_matriz(ruta, metricas, titulo):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    etiquetas = metricas["matriz_confusion"]["etiquetas"]
    matriz = np.asarray(metricas["matriz_confusion"]["matriz"], dtype=int)

    fig, ax = plt.subplots(figsize=(9, 7))
    imagen = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=35, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title(
        f"{titulo}\n"
        f"Accuracy={metricas['accuracy']:.4f} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"Sin activacion={metricas['sin_activacion']}"
    )

    umbral = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > umbral else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def guardar_histograma_puntajes(ruta, casos, titulo, minimo, maximo, corte, zona):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not casos:
        return

    puntajes = np.asarray([caso["puntaje"] for caso in casos], dtype=float)
    zona_min, zona_max, _ = zona
    dentro = int(np.sum((puntajes >= zona_min) & (puntajes <= zona_max)))
    porcentaje_dentro = (dentro / len(casos)) * 100.0
    puntaje_minimo_observado = float(np.min(puntajes))
    puntaje_maximo_observado = float(np.max(puntajes))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(puntajes, bins=16, range=(minimo, maximo), color="#4C78A8", edgecolor="white")
    ax.axvspan(zona_min, zona_max, color="#F58518", alpha=0.18, label=f"zona Wilson [{zona_min:.2f}, {zona_max:.2f}]")
    ax.axvline(zona_min, color="#2CA02C", linestyle="--", linewidth=2.0, label=f"limite IC inf={zona_min:.2f}")
    ax.axvline(zona_max, color="#2CA02C", linestyle="--", linewidth=2.0, label=f"limite IC sup={zona_max:.2f}")
    ax.axvline(corte, color="#D62728", linestyle="-", linewidth=1.8, label=f"corte={corte:.2f}")
    if minimo > 0.0:
        ax.axvline(minimo, color="#555555", linestyle="--", linewidth=1.5, label=f"min={minimo:.2f}")
    if maximo < 100.0:
        ax.axvline(maximo, color="#555555", linestyle="--", linewidth=1.5, label=f"max={maximo:.2f}")
    ax.set_title(
        f"{titulo}\n"
        f"casos={len(casos)} | dentro IC={dentro} ({porcentaje_dentro:.1f}%) | "
        f"min obs={puntaje_minimo_observado:.4f} | max obs={puntaje_maximo_observado:.4f}"
    )
    ax.set_xlabel("Puntaje de riesgo desfusificado")
    ax.set_ylabel("Cantidad de casos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    principal()
