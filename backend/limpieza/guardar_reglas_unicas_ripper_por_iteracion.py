"""Guarda las reglas unicas de RIPPER por cada iteracion experimental.

Uso desde backend:
    python limpieza/guardar_reglas_unicas_ripper_por_iteracion.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RAIZ_BACKEND = Path(__file__).resolve().parents[1]
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
    / "reglas_unicas_ripper_por_iteracion"
)


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_json(ruta: Path, contenido: dict | list) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clave_regla(regla: dict) -> tuple:
    antecedentes = regla.get("antecedentes", [])
    antecedentes_normalizados = tuple(
        (
            str(antecedente.get("variable")),
            str(antecedente.get("etiqueta_linguistica")),
        )
        for antecedente in antecedentes
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


def iteraciones_disponibles(ruta_resultados: Path, max_iteraciones: int) -> list[Path]:
    iteraciones = []
    for numero in range(1, max_iteraciones + 1):
        ruta = ruta_resultados / f"iteracion_{numero:02d}"
        if ruta.is_dir():
            iteraciones.append(ruta)
    return iteraciones


def construir_resultado_limpio(resultado: dict, reglas_unicas: list[dict], duplicadas: list[dict]) -> dict:
    limpio = dict(resultado)
    limpio["reglas_finales"] = reglas_unicas
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
        "criterio_duplicado": "mismos antecedentes en el mismo orden y mismo consecuente",
        "reglas_originales": len(resultado.get("reglas_finales", [])),
        "reglas_unicas": len(reglas_unicas),
        "reglas_eliminadas": len(duplicadas),
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
    resumen = []

    for ruta_iteracion in iteraciones_disponibles(ruta_resultados, max_iteraciones):
        ruta_json = ruta_iteracion / "ripper.json"
        if not ruta_json.exists():
            continue

        resultado = cargar_json(ruta_json)
        iteracion = int(resultado.get("iteracion", ruta_iteracion.name[-2:]))
        reglas_originales = resultado.get("reglas_finales", [])
        reglas_unicas, duplicadas = limpiar_reglas_iteracion(reglas_originales)
        resultado_limpio = construir_resultado_limpio(resultado, reglas_unicas, duplicadas)

        carpeta_iteracion = ruta_salida / f"iteracion_{iteracion:02d}"
        ruta_limpia = carpeta_iteracion / "ripper_reglas_unicas.json"
        guardar_json(ruta_limpia, resultado_limpio)

        conteo = contar_por_clase(reglas_unicas)
        resumen.append(
            {
                "iteracion": iteracion,
                "reglas_originales": len(reglas_originales),
                "reglas_unicas": len(reglas_unicas),
                "reglas_eliminadas": len(duplicadas),
                "low risk": conteo.get("low risk", 0),
                "mid risk": conteo.get("mid risk", 0),
                "high risk": conteo.get("high risk", 0),
                "balanced_accuracy_original": resultado.get("metricas", {}).get("balanced_accuracy"),
                "accuracy_original": resultado.get("metricas", {}).get("accuracy"),
                "ruta_origen": str(ruta_json),
                "ruta_limpia": str(ruta_limpia),
            }
        )

    salida = {
        "descripcion": "Reglas unicas de RIPPER guardadas por iteracion.",
        "criterio_duplicado": "mismos antecedentes en el mismo orden y mismo consecuente",
        "ruta_resultados": str(ruta_resultados),
        "ruta_salida": str(ruta_salida),
        "total_iteraciones": len(resumen),
        "iteraciones": resumen,
    }
    guardar_json(ruta_salida / "resumen_reglas_unicas_ripper_por_iteracion.json", salida)
    guardar_csv(ruta_salida / "resumen_reglas_unicas_ripper_por_iteracion.csv", resumen)
    return salida


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarda reglas unicas del ripper.json de cada iteracion."
    )
    parser.add_argument("--resultados", type=Path, default=RUTA_RESULTADOS_DEFAULT)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA_DEFAULT)
    parser.add_argument("--iteraciones", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    resumen = ejecutar(args.resultados, args.salida, args.iteraciones)

    print("Reglas unicas RIPPER guardadas por iteracion:")
    for fila in resumen["iteraciones"]:
        print(
            f"  iteracion_{fila['iteracion']:02d}: "
            f"{fila['reglas_originales']} -> {fila['reglas_unicas']} reglas "
            f"(eliminadas {fila['reglas_eliminadas']})"
        )
    print(f"Resumen: {args.salida / 'resumen_reglas_unicas_ripper_por_iteracion.json'}")


if __name__ == "__main__":
    main()
