"""
Ejecutar desde backend/:
    python mostrar_divisiones.py
"""

import pandas as pd
from pathlib import Path

RUTA_CSV = Path(__file__).parent / "src" / "riesgo_materno" / "datos" / "Maternal Health Risk Data Set.csv"

df = pd.read_csv(RUTA_CSV)
total = len(df)

print(f"\nTotal registros: {total}\n")
for clase, n in df["RiskLevel"].value_counts().items():
    print(f"  {clase:<12}  {n:>4}  ({n/total*100:.1f}%)")
print()
