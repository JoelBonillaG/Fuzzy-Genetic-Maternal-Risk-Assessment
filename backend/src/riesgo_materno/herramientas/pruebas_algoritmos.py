import argparse
import csv

from ..prediccion import predecir_caso


MAPA_RIESGO = {
    "low": "low risk",
    "mid": "mid risk",
    "high": "high risk",
}


def principal():
    argumentos = crear_parser().parse_args()

    if argumentos.csv:
        ejecutar_desde_csv(argumentos)
        return

    valores = recoger_valores(argumentos)
    resultado = predecir_caso(valores)
    texto = construir_salida_individual(resultado)
    print(texto)


def ejecutar_desde_csv(argumentos):
    filas = leer_filas_csv(argumentos.csv)
    filas_normalizadas = normalizar_filas_csv(filas)

    if argumentos.riesgo == "todos":
        riesgos_a_procesar = ["low risk", "mid risk", "high risk"]
    else:
        riesgos_a_procesar = [MAPA_RIESGO[argumentos.riesgo]]

    lineas_salida = []

    etiquetas = ["low risk", "mid risk", "high risk"]
    metricas = {
        riesgo_real: {
            "total": 0,
            "correctas": 0,
            "predicciones": {riesgo_predicho: 0 for riesgo_predicho in etiquetas},
        }
        for riesgo_real in etiquetas
    }

    for riesgo_objetivo in riesgos_a_procesar:
        filas_filtradas = [
            fila for fila in filas_normalizadas
            if fila["riesgo_csv"] == riesgo_objetivo
        ]

        if not filas_filtradas:
            lineas_salida.append(f"No se encontraron filas para: {riesgo_objetivo}")
            lineas_salida.append("-" * 70)
            continue

        cantidad = min(argumentos.cantidad, len(filas_filtradas))

        lineas_salida.append("=" * 70)
        lineas_salida.append(f"RIESGO OBJETIVO CSV: {riesgo_objetivo}")
        lineas_salida.append(f"Filas disponibles: {len(filas_filtradas)}")
        lineas_salida.append(f"Filas a ejecutar: {cantidad}")
        lineas_salida.append("=" * 70)

        for indice, fila in enumerate(filas_filtradas[:cantidad], start=1):
            valores = {
                "edad": fila["edad"],
                "presion_sistolica": fila["presion_sistolica"],
                "presion_diastolica": fila["presion_diastolica"],
                "azucar_sangre": fila["azucar_sangre"],
                "temperatura_corporal": fila["temperatura_corporal"],
                "frecuencia_cardiaca": fila["frecuencia_cardiaca"],
            }

            resultado = predecir_caso(valores)

            riesgo_real = fila["riesgo_csv"]
            riesgo_predicho = str(resultado["riesgo"]).strip().lower()

            if riesgo_real in metricas and riesgo_predicho in metricas[riesgo_real]["predicciones"]:
                metricas[riesgo_real]["total"] += 1
                metricas[riesgo_real]["predicciones"][riesgo_predicho] += 1
                if riesgo_real == riesgo_predicho:
                    metricas[riesgo_real]["correctas"] += 1

            lineas_salida.append(f"[{indice}/{cantidad}]")
            lineas_salida.append(construir_comando(valores))
            lineas_salida.append(f"Riesgo esperado CSV: {fila['riesgo_csv']}")
            lineas_salida.append(f"Sistema usado: {resultado['sistema']}")
            lineas_salida.append(f"Origen del modelo: {resultado['origen_modelo']}")

            for ajuste in resultado["ajustes_entrada"]:
                lineas_salida.append(
                    "Ajuste aplicado: "
                    f"{ajuste['variable']} "
                    f"{ajuste['valor_original']} -> {ajuste['valor_ajustado']}"
                )

            lineas_salida.append(f"Puntaje de riesgo: {resultado['puntaje']:.4f}")
            lineas_salida.append(f"Riesgo: {resultado['riesgo']}")
            lineas_salida.append("-" * 70)

    lineas_salida.append("")
    lineas_salida.append("=" * 70)
    lineas_salida.append("RESUMEN DE METRICAS")
    lineas_salida.append("=" * 70)

    total_global = 0
    aciertos_globales = 0

    for riesgo_real in riesgos_a_procesar:
        datos = metricas[riesgo_real]
        total = datos["total"]
        correctas = datos["correctas"]
        accuracy = (correctas / total * 100) if total > 0 else 0.0

        total_global += total
        aciertos_globales += correctas

        lineas_salida.append(f"Clase real: {riesgo_real}")
        lineas_salida.append(f"Total reales: {total}")
        lineas_salida.append(f"Predicciones correctas: {correctas}")
        lineas_salida.append(f"Accuracy por clase: {accuracy:.2f}%")

        for riesgo_predicho in etiquetas:
            cantidad_predicha = datos["predicciones"][riesgo_predicho]
            lineas_salida.append(
                f"  Real {riesgo_real} -> Predicho {riesgo_predicho}: {cantidad_predicha}"
            )

        lineas_salida.append("-" * 70)

    accuracy_global = (
        (aciertos_globales / total_global) * 100 if total_global > 0 else 0.0
    )

    lineas_salida.append("RESUMEN GLOBAL")
    lineas_salida.append(f"Total de casos evaluados: {total_global}")
    lineas_salida.append(f"Total de aciertos: {aciertos_globales}")
    lineas_salida.append(f"Accuracy global: {accuracy_global:.2f}%")
    lineas_salida.append("=" * 70)

    ruta_salida = argumentos.salida
    guardar_texto(ruta_salida, "\n".join(lineas_salida) + "\n")

    print(f"Resultados guardados en: {ruta_salida}")

def leer_filas_csv(ruta_csv):
    with open(ruta_csv, mode="r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        return list(lector)


def normalizar_filas_csv(filas):
    filas_normalizadas = []

    for fila in filas:
        riesgo_csv = str(fila["RiskLevel"]).strip().lower()

        filas_normalizadas.append({
            "edad": float(fila["Age"]),
            "presion_sistolica": float(fila["SystolicBP"]),
            "presion_diastolica": float(fila["DiastolicBP"]),
            "azucar_sangre": float(fila["BS"]),
            "temperatura_corporal": float(fila["BodyTemp"]),
            "frecuencia_cardiaca": float(fila["HeartRate"]),
            "riesgo_csv": riesgo_csv,
        })

    return filas_normalizadas


def recoger_valores(argumentos):
    return {
        "edad": argumentos.edad if argumentos.edad is not None else pedir_valor("edad"),
        "presion_sistolica": (
            argumentos.presion_sistolica
            if argumentos.presion_sistolica is not None
            else pedir_valor("presion_sistolica")
        ),
        "presion_diastolica": (
            argumentos.presion_diastolica
            if argumentos.presion_diastolica is not None
            else pedir_valor("presion_diastolica")
        ),
        "azucar_sangre": (
            argumentos.azucar_sangre
            if argumentos.azucar_sangre is not None
            else pedir_valor("azucar_sangre")
        ),
        "temperatura_corporal": (
            argumentos.temperatura_corporal
            if argumentos.temperatura_corporal is not None
            else pedir_valor("temperatura_corporal")
        ),
        "frecuencia_cardiaca": (
            argumentos.frecuencia_cardiaca
            if argumentos.frecuencia_cardiaca is not None
            else pedir_valor("frecuencia_cardiaca")
        ),
    }


def pedir_valor(nombre):
    return float(input(f"{nombre}: ").strip())


def construir_salida_individual(resultado):
    lineas = []
    lineas.append("Prediccion de riesgo materno")
    lineas.append("-" * 50)
    lineas.append(f"Sistema usado: {resultado['sistema']}")
    lineas.append(f"Origen del modelo: {resultado['origen_modelo']}")

    for ajuste in resultado["ajustes_entrada"]:
        lineas.append(
            "Ajuste aplicado: "
            f"{ajuste['variable']} "
            f"{ajuste['valor_original']} -> {ajuste['valor_ajustado']}"
        )

    lineas.append(f"Puntaje de riesgo: {resultado['puntaje']:.4f}")
    lineas.append(f"Riesgo: {resultado['riesgo']}")
    return "\n".join(lineas)


def construir_comando(valores):
    return (
        "venv\\Scripts\\python.exe -m riesgo_materno_difuso.herramientas.predecir_cli "
        f"--edad {formatear_numero(valores['edad'])} "
        f"--presion-sistolica {formatear_numero(valores['presion_sistolica'])} "
        f"--presion-diastolica {formatear_numero(valores['presion_diastolica'])} "
        f"--azucar-sangre {formatear_numero(valores['azucar_sangre'])} "
        f"--temperatura-corporal {formatear_numero(valores['temperatura_corporal'])} "
        f"--frecuencia-cardiaca {formatear_numero(valores['frecuencia_cardiaca'])}"
    )


def formatear_numero(valor):
    if float(valor).is_integer():
        return str(int(valor))
    return str(valor)


def guardar_texto(ruta_salida, contenido):
    with open(ruta_salida, mode="w", encoding="utf-8") as archivo:
        archivo.write(contenido)


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Predice el riesgo materno desde la linea de comandos."
    )

    parser.add_argument("--edad", type=float, help="Edad en anos.")
    parser.add_argument(
        "--presion-sistolica",
        dest="presion_sistolica",
        type=float,
        help="Presion sistolica en mmHg."
    )
    parser.add_argument(
        "--presion-diastolica",
        dest="presion_diastolica",
        type=float,
        help="Presion diastolica en mmHg."
    )
    parser.add_argument(
        "--azucar-sangre",
        dest="azucar_sangre",
        type=float,
        help="Azucar en sangre en mmol/L."
    )
    parser.add_argument(
        "--temperatura-corporal",
        dest="temperatura_corporal",
        type=float,
        help="Temperatura corporal en F."
    )
    parser.add_argument(
        "--frecuencia-cardiaca",
        dest="frecuencia_cardiaca",
        type=float,
        help="Frecuencia cardiaca en bpm."
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Ruta al CSV para ejecutar casos masivos."
    )
    parser.add_argument(
        "--riesgo",
        choices=["low", "mid", "high", "todos"],
        default="todos",
        help="Clase a extraer desde el CSV."
    )
    parser.add_argument(
        "--cantidad",
        type=int,
        default=100,
        help="Cantidad maxima de filas por riesgo a ejecutar."
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados_prediccion.txt",
        help="Archivo .txt donde se guardaran los resultados."
    )

    return parser


if __name__ == "__main__":
    principal()
