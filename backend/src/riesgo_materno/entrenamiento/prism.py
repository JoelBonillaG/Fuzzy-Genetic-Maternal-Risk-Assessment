"""Induccion de reglas PRISM sobre variables linguisticas discretizadas."""

from __future__ import annotations

import pandas as pd

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES, VARIABLES_ENTRADA
from .ripper import MAPA_CONSECUENTE, _discretizar


CLASES = ["low risk", "mid risk", "high risk"]


def aprender_reglas_prism(tabla, config):
    """Aprende reglas PRISM desde train_i y devuelve reglas para el motor difuso."""
    df = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    reglas = []
    for clase in config["class_order"]:
        positivos_restantes = set(df.index[df["riesgo"] == clase].tolist())
        reglas_clase = 0
        bloqueadas = set()
        while positivos_restantes and reglas_clase < config["max_rules_per_class"]:
            regla = construir_regla_prism(
                df=df,
                clase=clase,
                positivos_restantes=positivos_restantes,
                max_condiciones=config["max_conditions_per_rule"],
                bloqueadas=bloqueadas,
            )
            if regla is None:
                break
            clave = tuple(sorted(regla))
            bloqueadas.add(clave)
            cubiertos = indices_cubiertos(df, regla)
            positivos_cubiertos = {i for i in cubiertos if df.at[i, "riesgo"] == clase}
            if len(positivos_cubiertos) < config["min_rule_coverage"]:
                continue
            if config["remove_covered_positives"]:
                positivos_restantes -= positivos_cubiertos
            else:
                positivos_restantes -= {min(positivos_cubiertos)}
            reglas.append(
                {
                    "numero": len(reglas) + 1,
                    "antecedentes": regla,
                    "consecuente": MAPA_CONSECUENTE[clase],
                }
            )
            reglas_clase += 1
    return quitar_duplicadas(reglas)


def construir_regla_prism(df, clase, positivos_restantes, max_condiciones, bloqueadas):
    condiciones = []
    disponibles = {
        variable: list(ESPECIFICACIONES_VARIABLES[variable]["categorias"].keys())
        for variable in VARIABLES_ENTRADA
    }
    universo = set(df.index.tolist())

    for _ in range(max_condiciones):
        mejor = None
        for variable, categorias in disponibles.items():
            if any(variable == var for var, _ in condiciones):
                continue
            for categoria in categorias:
                candidata = condiciones + [(variable, categoria)]
                cubiertos = indices_cubiertos(df, candidata, universo)
                if not cubiertos:
                    continue
                positivos_restantes_cubiertos = len([i for i in cubiertos if i in positivos_restantes])
                positivos_clase = len([i for i in cubiertos if df.at[i, "riesgo"] == clase])
                precision = positivos_clase / len(cubiertos)
                score = (precision, positivos_restantes_cubiertos, positivos_clase, -len(cubiertos))
                if mejor is None or score > mejor[0]:
                    mejor = (score, (variable, categoria), cubiertos)
        if mejor is None or mejor[0][1] == 0:
            return condiciones or None
        condiciones.append(mejor[1])
        universo = mejor[2]
        if tuple(sorted(condiciones)) in bloqueadas:
            continue
        negativos = [i for i in universo if df.at[i, "riesgo"] != clase]
        if not negativos:
            return condiciones
    return condiciones or None


def indices_cubiertos(df, antecedentes, universo=None):
    indices = list(universo if universo is not None else df.index.tolist())
    for variable, categoria in antecedentes:
        indices = [i for i in indices if df.at[i, variable] == categoria]
    return set(indices)


def quitar_duplicadas(reglas):
    vistas = set()
    salida = []
    for regla in reglas:
        clave = (tuple(sorted(regla["antecedentes"])), regla["consecuente"])
        if clave in vistas:
            continue
        vistas.add(clave)
        regla = dict(regla)
        regla["numero"] = len(salida) + 1
        salida.append(regla)
    return salida
