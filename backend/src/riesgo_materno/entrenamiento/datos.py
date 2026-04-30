import pandas as pd
from sklearn.model_selection import train_test_split

from ..logica_difusa.variables import ETIQUETAS_RIESGO, VARIABLES_ENTRADA
from .modelo import (
    COLUMNA_RIESGO_CSV,
    MAPA_COLUMNAS_CSV,
    PROPORCION_ENTRENAMIENTO,
)


def cargar_dataset(ruta_csv):
    """Lee el CSV, renombra columnas al formato interno y valida que no falten datos."""
    tabla = pd.read_csv(ruta_csv)

    columnas_necesarias = list(MAPA_COLUMNAS_CSV.values()) + [COLUMNA_RIESGO_CSV]
    faltantes = []
    for columna in columnas_necesarias:
        if columna not in tabla.columns:
            faltantes.append(columna)

    if faltantes:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {faltantes}")

    datos = pd.DataFrame()
    for variable, columna_csv in MAPA_COLUMNAS_CSV.items():
        datos[variable] = pd.to_numeric(tabla[columna_csv], errors="coerce")

    datos["riesgo"] = tabla[COLUMNA_RIESGO_CSV].astype(str).str.strip().str.lower()

    etiquetas_desconocidas = []
    for etiqueta in datos["riesgo"].unique():
        if etiqueta not in ETIQUETAS_RIESGO:
            etiquetas_desconocidas.append(etiqueta)
    etiquetas_desconocidas = sorted(etiquetas_desconocidas)

    if etiquetas_desconocidas:
        raise ValueError(f"Hay clases no reconocidas en el CSV: {etiquetas_desconocidas}")
    if datos[VARIABLES_ENTRADA].isna().any().any():
        raise ValueError("El CSV contiene valores faltantes o no numericos.")

    datos = _quitar_filas_invalidas(datos)
    return datos


def dividir_entrenamiento_prueba(tabla, semilla=None):
    """Divide el dataset 70/30 estratificado por clase de riesgo."""
    entrenamiento, prueba = train_test_split(
        tabla,
        train_size=PROPORCION_ENTRENAMIENTO,
        stratify=tabla["riesgo"],
        shuffle=True,
        random_state=semilla,
    )
    return {
        "entrenamiento": entrenamiento.reset_index(drop=True),
        "prueba": prueba.reset_index(drop=True),
    }


def convertir_split_a_diccionario(tabla_split):
    """Convierte un DataFrame de split a dict {entradas: arrays por variable, riesgos: array de etiquetas}."""
    entradas = {}
    for variable in VARIABLES_ENTRADA:
        entradas[variable] = tabla_split[variable].to_numpy(dtype=float)

    riesgos = tabla_split["riesgo"].to_numpy(dtype=object)

    return {
        "entradas": entradas,
        "riesgos": riesgos,
    }


def resumir_splits(splits):
    """Genera una tabla con el tamaño y conteo por clase de cada split."""
    filas = []
    for nombre_split, tabla_split in splits.items():
        conteos = tabla_split["riesgo"].value_counts().reindex(ETIQUETAS_RIESGO, fill_value=0)
        filas.append(
            {
                "split": nombre_split,
                "tamano": len(tabla_split),
                "low risk": int(conteos["low risk"]),
                "mid risk": int(conteos["mid risk"]),
                "high risk": int(conteos["high risk"]),
            }
        )
    return pd.DataFrame(filas)


def _quitar_filas_invalidas(datos):
    # 2 filas con frecuencia cardiaca = 7 son errores del CSV original
    datos_limpios = datos.loc[datos["frecuencia_cardiaca"] != 7].copy()
    return datos_limpios.reset_index(drop=True)
