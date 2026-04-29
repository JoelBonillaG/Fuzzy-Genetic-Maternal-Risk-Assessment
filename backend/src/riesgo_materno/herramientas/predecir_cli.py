import argparse

from ..prediccion import predecir_caso


def principal():
    argumentos = crear_parser().parse_args()
    valores = recoger_valores(argumentos)
    resultado = predecir_caso(valores)

    print("Prediccion de riesgo materno")
    print("-" * 50)
    print(f"Sistema usado: {resultado['sistema']}")
    print(f"Origen del modelo: {resultado['origen_modelo']}")
    for ajuste in resultado["ajustes_entrada"]:
        print(
            "Ajuste aplicado: "
            f"{ajuste['variable']} "
            f"{ajuste['valor_original']} -> {ajuste['valor_ajustado']}"
        )
    print(f"Puntaje de riesgo: {resultado['puntaje']:.4f}")
    print(f"Riesgo: {resultado['riesgo']}")


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


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Predice el riesgo materno desde la linea de comandos."
    )
    parser.add_argument("--edad", type=float, help="Edad en anos.")
    parser.add_argument("--presion-sistolica", dest="presion_sistolica", type=float, help="Presion sistolica en mmHg.")
    parser.add_argument("--presion-diastolica", dest="presion_diastolica", type=float, help="Presion diastolica en mmHg.")
    parser.add_argument("--azucar-sangre", dest="azucar_sangre", type=float, help="Azucar en sangre en mmol/L.")
    parser.add_argument("--temperatura-corporal", dest="temperatura_corporal", type=float, help="Temperatura corporal en F.")
    parser.add_argument("--frecuencia-cardiaca", dest="frecuencia_cardiaca", type=float, help="Frecuencia cardiaca en bpm.")
    return parser


if __name__ == "__main__":
    principal()
