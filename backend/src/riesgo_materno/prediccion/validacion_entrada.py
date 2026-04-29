from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES, VARIABLES_ENTRADA


def construir_entrada_lote(valores_entrada):
    """Valida y convierte valores de entrada a formato de lote {variable: [valor]} para inferir_lote."""
    valores_validados, ajustes = validar_valores_entrada(valores_entrada)
    return (
        {variable: [valor] for variable, valor in valores_validados.items()},
        ajustes,
    )


def validar_valores_entrada(valores_entrada):
    """Valida cada variable contra su dominio, satura valores cercanos al limite y rechaza los que esten muy fuera."""
    valores_validados = {}
    ajustes = []
    for variable in VARIABLES_ENTRADA:
        valor = float(valores_entrada[variable])
        minimo, maximo = ESPECIFICACIONES_VARIABLES[variable]["limites"]
        tolerancia = ESPECIFICACIONES_VARIABLES[variable]["tolerancia_saturacion"]
        valor_ajustado = saturar_si_esta_cerca_del_limite(
            valor,
            minimo,
            maximo,
            tolerancia,
        )
        if valor_ajustado is None:
            raise ValueError(
                f"{variable}={valor} esta demasiado fuera del rango permitido [{minimo}, {maximo}]."
            )
        if valor_ajustado != valor:
            ajustes.append(
                {
                    "variable": variable,
                    "valor_original": valor,
                    "valor_ajustado": valor_ajustado,
                }
            )
        valores_validados[variable] = valor_ajustado
    return valores_validados, ajustes


def saturar_si_esta_cerca_del_limite(valor, minimo, maximo, tolerancia):
    """Satura al limite si el valor esta dentro de la tolerancia, devuelve None si esta demasiado fuera."""
    # Decision conservadora para entrada individual:
    # si el valor queda ligeramente fuera del dominio operativo del sistema,
    # se satura al limite; si se aleja demasiado, se rechaza.
    if minimo <= valor <= maximo:
        return valor
    if minimo - tolerancia <= valor < minimo:
        return minimo
    if maximo < valor <= maximo + tolerancia:
        return maximo
    return None
