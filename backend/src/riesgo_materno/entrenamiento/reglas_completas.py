"""Utilidades para forzar reglas difusas con todos los antecedentes."""

from __future__ import annotations

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES, VARIABLES_ENTRADA


def completar_antecedentes_reglas(reglas, df_discretizado, consecuente_a_clase):
    """Completa cada regla para que use las seis variables de entrada.

    Las reglas aprendidas por PRISM/RIPPER pueden ser parciales. Para cada
    variable faltante se escoge la categoria que mejora la precision sobre la
    clase de la regla dentro de los ejemplos cubiertos por sus antecedentes.
    """
    completas = []
    for regla in reglas:
        regla_completa = dict(regla)
        clase = consecuente_a_clase[regla["consecuente"]]
        antecedentes = dict(regla["antecedentes"])

        for variable in VARIABLES_ENTRADA:
            if variable in antecedentes:
                continue
            antecedentes[variable] = mejor_categoria_faltante(
                df_discretizado=df_discretizado,
                antecedentes_actuales=antecedentes,
                variable=variable,
                clase=clase,
            )

        regla_completa["antecedentes"] = [
            (variable, antecedentes[variable])
            for variable in VARIABLES_ENTRADA
        ]
        completas.append(regla_completa)
    return completas


def mejor_categoria_faltante(df_discretizado, antecedentes_actuales, variable, clase):
    universo = indices_cubiertos(df_discretizado, antecedentes_actuales.items())
    categorias = list(ESPECIFICACIONES_VARIABLES[variable]["categorias"].keys())

    mejor = None
    for categoria in categorias:
        cubiertos = [i for i in universo if df_discretizado.at[i, variable] == categoria]
        if not cubiertos:
            continue
        positivos = [i for i in cubiertos if df_discretizado.at[i, "riesgo"] == clase]
        score = (
            len(positivos) / len(cubiertos),
            len(positivos),
            len(cubiertos),
        )
        if mejor is None or score > mejor[0]:
            mejor = (score, categoria)

    if mejor is not None:
        return mejor[1]

    filas_clase = df_discretizado[df_discretizado["riesgo"] == clase]
    if not filas_clase.empty:
        return str(filas_clase[variable].mode().iloc[0])
    return str(df_discretizado[variable].mode().iloc[0])


def indices_cubiertos(df_discretizado, antecedentes):
    indices = df_discretizado.index.tolist()
    for variable, categoria in antecedentes:
        indices = [i for i in indices if df_discretizado.at[i, variable] == categoria]
    return indices
