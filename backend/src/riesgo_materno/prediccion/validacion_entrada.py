from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES, VARIABLES_ENTRADA


def construir_entrada_lote(valores_entrada):
    """Valida y convierte valores de entrada a formato de lote {variable: [valor]} para inferir_lote."""
    valores_validados, ajustes = validar_valores_entrada(valores_entrada)
    return (
        {variable: [valor] for variable, valor in valores_validados.items()},
        ajustes,
    )


def validar_valores_entrada(valores_entrada):
    """Valida que cada variable este dentro de su dominio definido en limites."""
    valores_validados = {}
    for variable in VARIABLES_ENTRADA:
        valor = float(valores_entrada[variable])
        minimo, maximo = ESPECIFICACIONES_VARIABLES[variable]["limites"]
        if not (minimo <= valor <= maximo):
            raise ValueError(
                f"{variable}={valor} fuera del rango permitido [{minimo}, {maximo}]."
            )
        valores_validados[variable] = valor
    return valores_validados, []
