from collections import OrderedDict
from pathlib import Path


RUTA_PAQUETE = Path(__file__).resolve().parents[1]
RUTA_CSV = RUTA_PAQUETE / "datos" / "Maternal Health Risk Data Set.csv"
COLUMNA_RIESGO_CSV = "RiskLevel"
RUTA_MODELO_OPTIMIZADO  = RUTA_PAQUETE / "modelos" / "modelo_optimizado.json"
RUTA_REGLAS_APRENDIDAS  = RUTA_PAQUETE / "modelos" / "reglas_aprendidas.json"

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

PARAMETROS_AG = {
    # cuantos cromosomas hay en cada generacion
    "tamano_poblacion": 50,
    # cuantos hijos se generan por generacion via cruce
    "cantidad_hijos": 50,
    # tope de generaciones si no hay parada temprana
    "maximo_generaciones": 60,
    # probabilidad de que dos padres se crucen para generar hijos
    "probabilidad_cruce": 0.85,
    # probabilidad de que cada gen individual mute
    "probabilidad_mutacion": 0.04,
    # cuantos mejores individuos pasan directamente a la siguiente generacion sin cambios
    "elitismo": 3,
    # cuantos individuos compiten en cada seleccion por torneo
    "tamano_torneo": 3,
    # generaciones sin mejora antes de detener el AG
    "paciencia": 50,
}

PESOS_FITNESS = {
    # macro_f1: peso principal — clasifica bien las 3 clases por igual
    "macro_f1": 0.75,
    # recall_alto: peso extra — detectar riesgo alto es prioritario (falso negativo = consecuencia grave)
    "recall_alto": 0.25,
    # interpretabilidad: penaliza huecos y solapamiento excesivo — el medico debe poder entender las funciones
    "interpretabilidad": 0.8,
    # desviacion: penaliza alejarse demasiado del cromosoma base — guia suave para no perder significado clinico
    "desviacion": 0.05,
}

PROPORCIONES_SPLIT = {
    "entrenamiento": 0.70,
    "validacion": 0.15,
    "prueba": 0.15,
}
