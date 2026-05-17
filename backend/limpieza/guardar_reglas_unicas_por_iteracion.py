"""Guarda las reglas unicas del AG por cada iteracion experimental.

Uso desde backend:
    python limpieza/guardar_reglas_unicas_por_iteracion.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


RAIZ_BACKEND = Path(__file__).resolve().parents[1]
RUTA_SRC = RAIZ_BACKEND / "src"
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

from riesgo_materno.entrenamiento.datos import cargar_dataset
from riesgo_materno.entrenamiento.modelo import RUTA_CSV
from riesgo_materno.herramientas.comparaciones.experimento_reglas import evaluar_reglas


RUTA_RESULTADOS_DEFAULT = (
    RAIZ_BACKEND
    / "src"
    / "riesgo_materno"
    / "herramientas"
    / "comparaciones"
    / "resultados"
)
RUTA_SALIDA_DEFAULT = (
    Path(__file__).resolve().parent
    / "resultados_limpieza"
    / "reglas_unicas_por_iteracion"
)
CLASE_A_CONSECUENTE = {
    "low risk": "bajo",
    "mid risk": "medio",
    "high risk": "alto",
    "bajo": "bajo",
    "medio": "medio",
    "alto": "alto",
}


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def guardar_json(ruta: Path, contenido: dict | list) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalizar_antecedente(antecedente) -> tuple[str, str]:
    if isinstance(antecedente, dict):
        return (
            str(antecedente.get("variable")),
            str(antecedente.get("etiqueta_linguistica")),
        )
    variable, etiqueta = antecedente
    return str(variable), str(etiqueta)


def clave_regla(regla: dict) -> tuple:
    antecedentes = regla.get("antecedentes", [])
    antecedentes_normalizados = tuple(
        normalizar_antecedente(antecedente) for antecedente in antecedentes
    )
    return antecedentes_normalizados, str(regla.get("consecuente"))


def contar_por_clase(reglas: list[dict]) -> dict:
    conteo = Counter(str(regla.get("consecuente")) for regla in reglas)
    return dict(sorted(conteo.items()))


def renumerar_reglas(reglas: list[dict]) -> list[dict]:
    renumeradas = []
    for indice, regla in enumerate(reglas, start=1):
        copia = dict(regla)
        copia["id"] = f"R{indice:03d}"
        renumeradas.append(copia)
    return renumeradas


def limpiar_reglas_iteracion(reglas: list[dict]) -> tuple[list[dict], list[dict]]:
    vistas = set()
    unicas = []
    duplicadas = []

    for regla in reglas:
        clave = clave_regla(regla)
        if clave in vistas:
            duplicadas.append(regla)
            continue
        vistas.add(clave)
        unicas.append(regla)

    return renumerar_reglas(unicas), duplicadas


def convertir_regla_a_motor(regla: dict) -> dict:
    consecuente = str(regla.get("consecuente"))
    if consecuente not in CLASE_A_CONSECUENTE:
        raise ValueError(f"Consecuente no reconocido: {consecuente!r}")
    return {
        "numero": regla.get("id"),
        "antecedentes": [
            normalizar_antecedente(antecedente)
            for antecedente in regla.get("antecedentes", [])
        ],
        "consecuente": CLASE_A_CONSECUENTE[consecuente],
        "source": regla.get("origen", "AG_MICHIGAN_BINARIO"),
    }


def evaluar_reglas_unicas(reglas_unicas: list[dict], tabla) -> tuple[dict, dict]:
    reglas_motor = [convertir_regla_a_motor(regla) for regla in reglas_unicas]
    metricas = evaluar_reglas(reglas_motor, tabla)
    matriz = metricas.pop("matriz_confusion")
    return metricas, matriz


def iteraciones_disponibles(ruta_resultados: Path, max_iteraciones: int) -> list[Path]:
    iteraciones = []
    for numero in range(1, max_iteraciones + 1):
        ruta = ruta_resultados / f"iteracion_{numero:02d}"
        if ruta.is_dir():
            iteraciones.append(ruta)
    return iteraciones


def construir_resultado_limpio(
    resultado: dict,
    reglas_unicas: list[dict],
    duplicadas: list[dict],
    metricas_limpias: dict,
    matriz_limpia: dict,
) -> dict:
    limpio = dict(resultado)
    limpio["reglas_finales"] = reglas_unicas
    limpio["metricas_originales_antes_limpieza"] = dict(resultado.get("metricas", {}))
    limpio["matriz_confusion_original_antes_limpieza"] = dict(resultado.get("matriz_confusion", {}))
    limpio["metricas"] = metricas_limpias
    limpio["matriz_confusion"] = matriz_limpia
    limpio["resumen_reglas"] = {
        **dict(resultado.get("resumen_reglas", {})),
        "total_reglas_original": len(resultado.get("reglas_finales", [])),
        "total_reglas": len(reglas_unicas),
        "reglas_duplicadas_original": len(duplicadas),
        "reglas_duplicadas": 0,
        "reglas_eliminadas": len(duplicadas),
        "reglas_por_clase": contar_por_clase(reglas_unicas),
    }
    limpio["limpieza"] = {
        "criterio_duplicado": "mismos antecedentes en el mismo orden dentro de la misma clase/consecuente",
        "reglas_originales": len(resultado.get("reglas_finales", [])),
        "reglas_unicas": len(reglas_unicas),
        "reglas_eliminadas": len(duplicadas),
        "metricas_recalculadas_con_reglas_limpias": True,
    }
    return limpio


def guardar_csv(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(filas[0].keys()) if filas else []
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def ejecutar(ruta_resultados: Path, ruta_salida: Path, max_iteraciones: int) -> dict:
    ruta_salida.mkdir(parents=True, exist_ok=True)
    tabla = cargar_dataset(RUTA_CSV)
    resumen = []

    for ruta_iteracion in iteraciones_disponibles(ruta_resultados, max_iteraciones):
        ruta_json = ruta_iteracion / "genetic_algorithm.json"
        if not ruta_json.exists():
            continue

        resultado = cargar_json(ruta_json)
        iteracion = int(resultado.get("iteracion", ruta_iteracion.name[-2:]))
        reglas_originales = resultado.get("reglas_finales", [])
        reglas_unicas, duplicadas = limpiar_reglas_iteracion(reglas_originales)
        metricas_limpias, matriz_limpia = evaluar_reglas_unicas(reglas_unicas, tabla)
        resultado_limpio = construir_resultado_limpio(
            resultado,
            reglas_unicas,
            duplicadas,
            metricas_limpias,
            matriz_limpia,
        )

        carpeta_iteracion = ruta_salida / f"iteracion_{iteracion:02d}"
        ruta_limpia = carpeta_iteracion / "genetic_algorithm_reglas_unicas.json"
        guardar_json(ruta_limpia, resultado_limpio)

        reglas_por_clase = contar_por_clase(reglas_unicas)
        resumen.append(
            {
                "iteracion": iteracion,
                "reglas_originales": len(reglas_originales),
                "reglas_unicas": len(reglas_unicas),
                "reglas_eliminadas": len(duplicadas),
                "low risk": reglas_por_clase.get("low risk", 0),
                "mid risk": reglas_por_clase.get("mid risk", 0),
                "high risk": reglas_por_clase.get("high risk", 0),
                "balanced_accuracy_original": resultado.get("metricas", {}).get("balanced_accuracy"),
                "accuracy_original": resultado.get("metricas", {}).get("accuracy"),
                "balanced_accuracy_limpia": metricas_limpias.get("balanced_accuracy"),
                "accuracy_limpia": metricas_limpias.get("accuracy"),
                "sin_activacion_limpia": metricas_limpias.get("sin_activacion"),
                "ruta_origen": str(ruta_json),
                "ruta_limpia": str(ruta_limpia),
            }
        )

    mejor_por_ba = max(resumen, key=lambda fila: fila["balanced_accuracy_limpia"]) if resumen else None
    mejor_por_accuracy = max(resumen, key=lambda fila: fila["accuracy_limpia"]) if resumen else None
    salida = {
        "descripcion": "Reglas unicas del AG Michigan binario guardadas por iteracion.",
        "criterio_duplicado": "mismos antecedentes en el mismo orden dentro de la misma clase/consecuente",
        "ruta_resultados": str(ruta_resultados),
        "ruta_salida": str(ruta_salida),
        "total_iteraciones": len(resumen),
        "mejor_por_balanced_accuracy_limpia": mejor_por_ba,
        "mejor_por_accuracy_limpia": mejor_por_accuracy,
        "iteraciones": resumen,
    }
    guardar_json(ruta_salida / "resumen_reglas_unicas_por_iteracion.json", salida)
    guardar_csv(ruta_salida / "resumen_reglas_unicas_por_iteracion.csv", resumen)
    return salida


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarda reglas unicas del genetic_algorithm.json de cada iteracion."
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RUTA_RESULTADOS_DEFAULT,
        help="Carpeta que contiene iteracion_01 ... iteracion_20.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_SALIDA_DEFAULT,
        help="Carpeta donde se guardaran las reglas unicas por iteracion.",
    )
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=20,
        help="Cantidad maxima de iteraciones a revisar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    resumen = ejecutar(args.resultados, args.salida, args.iteraciones)

    print("Reglas unicas guardadas por iteracion:")
    for fila in resumen["iteraciones"]:
        print(
            f"  iteracion_{fila['iteracion']:02d}: "
            f"{fila['reglas_originales']} -> {fila['reglas_unicas']} reglas "
            f"(eliminadas {fila['reglas_eliminadas']})"
        )
    print(f"Resumen: {args.salida / 'resumen_reglas_unicas_por_iteracion.json'}")


if __name__ == "__main__":
    main()
