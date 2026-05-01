from collections import OrderedDict
from pathlib import Path


RUTA_PAQUETE = Path(__file__).resolve().parents[1]
RUTA_CSV = RUTA_PAQUETE / "datos" / "Maternal Health Risk Data Set.csv"
COLUMNA_RIESGO_CSV = "RiskLevel"
RUTA_REGLAS_APRENDIDAS = RUTA_PAQUETE / "modelos" / "reglas_aprendidas.json"
RUTA_SELECCION_REGLAS = RUTA_PAQUETE / "modelos" / "modelo_optimizado_reglas.json"

MAPA_COLUMNAS_CSV = OrderedDict(
    [
        ("edad", "Age"),
        ("presion_sistolica", "SystolicBP"),
        ("presion_diastolica", "DiastolicBP"),
        ("azucar_sangre", "BS"),
        ("temperatura_corporal", "BodyTemp"),
        ("frecuencia_cardiaca", "HeartRate"),
    ]
)

# 70/30 estratificado entrenamiento/prueba.
PROPORCION_ENTRENAMIENTO = 0.70

PARAMETROS_AG = {
    # cuantos cromosomas binarios hay en cada generacion
    "tamano_poblacion": 50,
    # cuantos padres se seleccionan por ruleta para entrar al pool de cruce
    # PyGAD luego cruza pares de este pool hasta producir (tamano_poblacion - elitismo) hijos
    "cantidad_padres": 25,
    # tope de generaciones
    "maximo_generaciones": 50,
    # probabilidad de aplicar cruce de un punto entre dos padres
    "probabilidad_cruce": 0.85,
    # probabilidad de que cada bit individual se invierta  (flip)
    "probabilidad_mutacion": 0.05,
    # cuantos mejores individuos pasan intactos a la siguiente generacion
    "elitismo": 1,
    # generaciones sin mejora antes de detener el AG
    "paciencia": 30,
}

# Pesos del fitness Pittsburgh: Fitness(S) = w_ba * BA(S) - w_compacidad * C(S)
#   BA(S)  = Balanced Accuracy — promedio del recall por clase (trata todas las clases igual)
#   C(S)   = |S| / |Scand|     — fraccion de reglas candidatas seleccionadas (0 a 1)
PESOS_FITNESS = {
    "balanced_accuracy": 0.95,
    "compacidad": 0.05,
}
