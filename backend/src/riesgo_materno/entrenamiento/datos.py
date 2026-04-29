import pandas as pd
from sklearn.model_selection import train_test_split

from ..logica_difusa.variables import ETIQUETAS_RIESGO, VARIABLES_ENTRADA
from .modelo import (
    COLUMNA_RIESGO_CSV,
    MAPA_COLUMNAS_CSV,
    PROPORCIONES_SPLIT,
)


def cargar_datos(ruta_csv):
    """Lee el CSV, renombra columnas al formato interno y valida que no falten datos."""
    tabla = pd.read_csv(ruta_csv)
    columnas_necesarias = list(MAPA_COLUMNAS_CSV.values()) + [COLUMNA_RIESGO_CSV]
    faltantes = [columna for columna in columnas_necesarias if columna not in tabla.columns]

    if faltantes:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {faltantes}")

    datos = pd.DataFrame()
    for variable, columna_csv in MAPA_COLUMNAS_CSV.items():
        datos[variable] = pd.to_numeric(tabla[columna_csv], errors="coerce")

    datos["riesgo"] = tabla[COLUMNA_RIESGO_CSV].astype(str).str.strip().str.lower()
    etiquetas_desconocidas = sorted(set(datos["riesgo"].unique()) - set(ETIQUETAS_RIESGO))

    if etiquetas_desconocidas:
        raise ValueError(f"Hay clases no reconocidas en el CSV: {etiquetas_desconocidas}")
    if datos[VARIABLES_ENTRADA].isna().any().any():
        raise ValueError("El CSV contiene valores faltantes o no numericos.")

    datos = quitar_registros_con_frecuencia_cardiaca_erronea(datos)
    return datos


def dividir_datos_estratificados(tabla):
    """Divide el dataset en entrenamiento/validacion/prueba manteniendo proporciones de clase en cada split."""
    validar_proporciones()
    proporcion_temporal = PROPORCIONES_SPLIT["validacion"] + PROPORCIONES_SPLIT["prueba"]

    entrenamiento, temporal = train_test_split(
        tabla,
        train_size=PROPORCIONES_SPLIT["entrenamiento"],
        stratify=tabla["riesgo"],
        shuffle=True,
    )
    validacion, prueba = train_test_split(
        temporal,
        train_size=PROPORCIONES_SPLIT["validacion"] / proporcion_temporal,
        stratify=temporal["riesgo"],
        shuffle=True,
    )

    return {
        "entrenamiento": entrenamiento.reset_index(drop=True),
        "validacion": validacion.reset_index(drop=True),
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


def validar_proporciones():
    """Verifica que las proporciones de split sumen exactamente 1.0."""
    total = sum(PROPORCIONES_SPLIT.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Las proporciones de entrenamiento, validacion y prueba deben sumar 1.")


def quitar_registros_con_frecuencia_cardiaca_erronea(datos):
    """Elimina las 2 filas del dataset con frecuencia cardiaca=7, valor erroneo del CSV original."""
    # Limpieza puntual del dataset original:
    # existen 2 filas con frecuencia cardiaca igual a 7, valor que no debe
    # participar en el entrenamiento ni en la optimizacion.
    datos_limpios = datos.loc[datos["frecuencia_cardiaca"] != 7].copy()
    return datos_limpios.reset_index(drop=True)
