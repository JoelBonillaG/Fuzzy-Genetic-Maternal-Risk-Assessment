"""Induccion de reglas incompletas con RIPPER."""

import pandas as pd
from wittgenstein import RIPPER

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES


ORDEN_CLASES = ["high risk", "mid risk", "low risk"]
MAPA_CONSECUENTE = {
    "high risk": "alto",
    "mid risk": "medio",
    "low risk": "bajo",
}


def _grado_trapecio(x, puntos):
    """Calcula el grado de pertenencia de x en un trapecio [a, b, c, d]."""
    a, b, c, d = puntos
    if x < a or x > d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 0.0


def _categoria(valor, categorias):
    """Devuelve la categoria con mayor pertenencia."""
    return max(categorias, key=lambda cat: _grado_trapecio(valor, categorias[cat]))


def _discretizar(tabla):
    """Convierte el dataset numerico en categorias linguisticas."""
    ejemplos = []
    for _, fila in tabla.iterrows():
        ejemplo = {
            variable: _categoria(fila[variable], spec["categorias"])
            for variable, spec in ESPECIFICACIONES_VARIABLES.items()
        }
        ejemplo["clase"] = fila["riesgo"]
        ejemplos.append(ejemplo)
    return ejemplos


def aprender_reglas_ripper(tabla, orden_clases=None, parametros=None):
    """Aprende reglas IF-THEN por clase usando RIPPER."""
    df = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    orden = orden_clases or ORDEN_CLASES
    parametros = parametros or {}
    reglas = []

    for clase in orden:
        df_binario = df.copy()
        df_binario["riesgo"] = df["riesgo"].apply(lambda x: clase if x == clase else "otro")

        clasificador = RIPPER(**parametros)
        clasificador.fit(df_binario, class_feat="riesgo", pos_class=clase)

        for rule in clasificador.ruleset_.rules:
            condiciones = [(cond.feature, cond.val) for cond in rule.conds]
            if not condiciones:
                continue
            reglas.append(
                {
                    "numero": len(reglas) + 1,
                    "antecedentes": condiciones,
                    "consecuente": MAPA_CONSECUENTE[clase],
                }
            )

    return reglas


def evaluar_reglas_duras(reglas, tabla):
    """Evalua reglas discretas por primera coincidencia."""
    ejemplos = _discretizar(tabla)
    aciertos = 0

    for ejemplo in ejemplos:
        prediccion = None
        for regla in reglas:
            if all(ejemplo[var] == cat for var, cat in regla["antecedentes"]):
                prediccion = {"alto": "high risk", "medio": "mid risk", "bajo": "low risk"}[
                    regla["consecuente"]
                ]
                break
        if prediccion == ejemplo["clase"]:
            aciertos += 1

    return aciertos / len(ejemplos)
