from collections import OrderedDict
from pathlib import Path


RUTA_PAQUETE = Path(__file__).resolve().parents[1]
RUTA_CSV = RUTA_PAQUETE / "datos" / "Maternal Health Risk Data Set.csv"
COLUMNA_RIESGO_CSV = "RiskLevel"
RUTA_REGLAS = RUTA_PAQUETE / "reglas"
RUTA_REGLAS_LIMPIAS = RUTA_REGLAS / "experimentos"
RUTA_REGLAS_SISTEMA_DIFUSO = RUTA_REGLAS / "reglas_sistema_difuso.json"
RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER = RUTA_REGLAS / "reglas_sistema_difuso_ripper.json"
RUTA_METADATA_REGLAS_SISTEMA_DIFUSO = RUTA_REGLAS / "metadata_reglas_sistema_difuso.json"

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
    "tamano_poblacion": 150,
    "cantidad_padres": 50,
    "maximo_generaciones": 500,
    "probabilidad_cruce": 0.90,
    "probabilidad_mutacion": 0.06,
    "elitismo": 4,
    "paciencia": 50,
}
