"""
Identifica observaciones con valores identicos en todas las variables
pero con etiquetas de riesgo distintas (conflictos de clase).

Genera dos archivos en la misma carpeta:
  - conflictos_resumen.csv  : un grupo por combinacion conflictiva
  - conflictos_detalle.csv  : todas las filas originales involucradas
"""

import pandas as pd
from pathlib import Path

RUTA_CSV = Path(__file__).parent / "Maternal Health Risk Data Set.csv"
RUTA_RESUMEN = Path(__file__).parent / "conflictos_resumen.csv"
RUTA_DETALLE = Path(__file__).parent / "conflictos_detalle.csv"

FEATURES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

def main():
    df = pd.read_csv(RUTA_CSV)
    df = df[df["HeartRate"] != 7].reset_index(drop=True)

    grupos = df.groupby(FEATURES)["RiskLevel"].agg(
        clases_distintas="nunique",
        clases_lista=list,
        total_casos="count",
    ).reset_index()

    conflictos = grupos[grupos["clases_distintas"] > 1].copy()
    conflictos = conflictos.sort_values("total_casos", ascending=False)

    print("=" * 60)
    print("CONFLICTOS DE CLASE EN EL DATASET")
    print("=" * 60)
    print(f"Total grupos conflictivos:      {len(conflictos)}")
    print(f"Observaciones en conflicto:     {conflictos['total_casos'].sum()}")
    print(f"Total dataset (limpio):         {len(df)}")
    print(f"Porcentaje conflicto:           {100 * conflictos['total_casos'].sum() / len(df):.1f}%")
    print()

    print("TOP 10 GRUPOS MAS FRECUENTES:")
    print("-" * 60)
    for _, fila in conflictos.head(10).iterrows():
        print(
            f"  Age={int(fila['Age'])} SBP={int(fila['SystolicBP'])} DBP={int(fila['DiastolicBP'])} "
            f"BS={fila['BS']} T={fila['BodyTemp']} HR={int(fila['HeartRate'])}"
        )
        conteo = pd.Series(fila["clases_lista"]).value_counts().to_dict()
        for clase, n in conteo.items():
            print(f"    {clase}: {n} vez/veces")
        print()

    print("DISTRIBUCION DE CONFLICTOS POR PAR DE CLASES:")
    print("-" * 60)
    pares = {"low-mid": 0, "mid-high": 0, "low-high": 0, "los tres": 0}
    for _, fila in conflictos.iterrows():
        clases = set(fila["clases_lista"])
        if clases == {"low risk", "mid risk"}:
            pares["low-mid"] += fila["total_casos"]
        elif clases == {"mid risk", "high risk"}:
            pares["mid-high"] += fila["total_casos"]
        elif clases == {"low risk", "high risk"}:
            pares["low-high"] += fila["total_casos"]
        else:
            pares["los tres"] += fila["total_casos"]
    for par, n in pares.items():
        print(f"  {par}: {n} observaciones")

    conflictos_export = conflictos.copy()
    conflictos_export["clases_lista"] = conflictos_export["clases_lista"].apply(
        lambda x: " | ".join(sorted(set(x)))
    )
    conflictos_export.to_csv(RUTA_RESUMEN, index=False)
    print(f"\nResumen guardado en: {RUTA_RESUMEN.name}")

    filas_conflicto = df.merge(
        conflictos[FEATURES], on=FEATURES, how="inner"
    ).sort_values(FEATURES)
    filas_conflicto.to_csv(RUTA_DETALLE, index=False)
    print(f"Detalle guardado en: {RUTA_DETALLE.name}")


if __name__ == "__main__":
    main()
