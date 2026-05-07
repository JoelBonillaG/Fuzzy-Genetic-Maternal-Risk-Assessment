from collections import OrderedDict


VARIABLES_ENTRADA = [
    "edad",
    "presion_sistolica",
    "presion_diastolica",
    "azucar_sangre",
    "temperatura_corporal",
    "frecuencia_cardiaca",
]

ETIQUETAS_RIESGO = ["low risk", "mid risk", "high risk"]

SALIDA_DIFUSA = {
    "nombre": "puntaje_riesgo",
    "universo": (0.0, 100.0),
    "categorias": OrderedDict(
        [
            ("bajo", [0.0, 0.0, 25.0, 40.0]),
            ("medio", [35.0, 45.0, 58.0, 70.0]),
            ("alto", [62.0, 75.0, 100.0, 100.0]),
        ]
    ),
}

ESPECIFICACIONES_VARIABLES = OrderedDict(
    [
        (
            "edad",
            {
                "limites": (10.0, 70.0),
                "categorias": OrderedDict(
                    [
                        ("adolescente", [10.0, 10.0, 17.0, 20.0]),
                        ("optima", [18.0, 20.0, 32.0, 35.0]),
                        ("avanzada", [33.0, 35.0, 39.0, 41.0]),
                        ("muy_avanzada", [39.0, 41.0, 55.0, 70.0]),
                    ]
                ),
            },
        ),
        (
            "presion_sistolica",
            {
                "limites": (70.0, 220.0),
                "categorias": OrderedDict(
                    [
                        ("hipotension", [70.0, 70.0, 85.0, 95.0]),
                        ("normal", [90.0, 100.0, 119.0, 125.0]),
                        ("elevada", [120.0, 128.0, 138.0, 142.0]),
                        ("hipertension", [140.0, 145.0, 158.0, 162.0]),
                        ("hipertension_severa", [160.0, 165.0, 220.0, 220.0]),
                    ]
                ),
            },
        ),
        (
            "presion_diastolica",
            {
                "limites": (49.0, 130.0),
                "categorias": OrderedDict(
                    [
                        ("normal", [49.0, 55.0, 79.0, 85.0]),
                        ("elevada", [80.0, 85.0, 89.0, 92.0]),
                        ("hipertension", [90.0, 95.0, 108.0, 112.0]),
                        ("hipertension_severa", [110.0, 113.0, 130.0, 130.0]),
                    ]
                ),
            },
        ),
        (
            "azucar_sangre",
            {
                "limites": (6.0, 19.0),
                "categorias": OrderedDict(
                    [
                        ("normoglucemia", [6.0, 6.0, 7.0, 7.8]),
                        ("hiperglucemia_gestacional", [7.5, 8.0, 10.5, 11.5]),
                        ("diabetes_manifiesta", [11.0, 12.0, 19.0, 19.0]),
                    ]
                ),
            },
        ),
        (
            "temperatura_corporal",
            {
                "limites": (95.0, 105.0),
                "categorias": OrderedDict(
                    [
                        ("normal", [95.0, 96.8, 99.0, 99.5]),
                        ("febricular", [99.0, 99.5, 100.2, 100.5]),
                        ("fiebre", [100.4, 100.8, 102.0, 102.5]),
                        ("hiperpirexia", [102.2, 102.5, 105.0, 105.0]),
                    ]
                ),
            },
        ),
        (
            "frecuencia_cardiaca",
            {
                "limites": (40.0, 160.0),
                "categorias": OrderedDict(
                    [
                        ("bradicardia", [40.0, 40.0, 58.0, 65.0]),
                        ("normal", [60.0, 70.0, 95.0, 100.0]),
                        ("taquicardia", [95.0, 100.0, 160.0, 160.0]),
                    ]
                ),
            },
        ),
    ]
)

# Eje x sobre el que se construye el area agregada en la desfusificacion
PUNTOS_SALIDA = 401

# Eje x sobre el que se dibuja la curva trapezoidal
PUNTOS_GRAFICA = 300
