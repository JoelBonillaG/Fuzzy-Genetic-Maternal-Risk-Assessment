import pandas as pd

from ..logica_difusa.variables import ETIQUETAS_RIESGO, VARIABLES_ENTRADA
from .modelo import COLUMNA_RIESGO_CSV, MAPA_COLUMNAS_CSV


def cargar_dataset(ruta_csv):
    """Lee el CSV, renombra columnas al formato interno y valida los datos."""
    tabla = pd.read_csv(ruta_csv)

    columnas_necesarias = list(MAPA_COLUMNAS_CSV.values()) + [COLUMNA_RIESGO_CSV]
    faltantes = [columna for columna in columnas_necesarias if columna not in tabla.columns]
    if faltantes:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {faltantes}")

    datos = pd.DataFrame()
    for variable, columna_csv in MAPA_COLUMNAS_CSV.items():
        datos[variable] = pd.to_numeric(tabla[columna_csv], errors="coerce")

    datos["riesgo"] = tabla[COLUMNA_RIESGO_CSV].astype(str).str.strip().str.lower()

    etiquetas_desconocidas = sorted(
        etiqueta for etiqueta in datos["riesgo"].unique() if etiqueta not in ETIQUETAS_RIESGO
    )
    if etiquetas_desconocidas:
        raise ValueError(f"Hay clases no reconocidas en el CSV: {etiquetas_desconocidas}")

    if datos[VARIABLES_ENTRADA].isna().any().any():
        raise ValueError("El CSV contiene valores faltantes o no numericos.")

    return _quitar_filas_invalidas(datos)


def convertir_split_a_diccionario(tabla_split):
    """Convierte un DataFrame a dict {entradas: arrays por variable, riesgos: array}."""
    entradas = {
        variable: tabla_split[variable].to_numpy(dtype=float)
        for variable in VARIABLES_ENTRADA
    }
    return {
        "entradas": entradas,
        "riesgos": tabla_split["riesgo"].to_numpy(dtype=object),
    }


def _quitar_filas_invalidas(datos):
    # 2 filas con frecuencia cardiaca = 7 son errores del CSV original.
    datos_limpios = datos.loc[datos["frecuencia_cardiaca"] != 7].copy()
    return datos_limpios.reset_index(drop=True)
