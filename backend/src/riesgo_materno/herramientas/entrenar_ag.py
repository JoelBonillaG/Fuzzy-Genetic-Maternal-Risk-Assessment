"""CLI de entrenamiento: corre el AG Pittsburgh y guarda la seleccion de reglas.

Uso:
    python -m src.riesgo_materno.herramientas.entrenar_ag
"""

from ..entrenamiento.entrenador import entrenar_seleccion_reglas


def principal():
    print("=" * 70)
    print("Entrenamiento AG Pittsburgh - seleccion de reglas difusas")
    print("=" * 70)

    resultado = entrenar_seleccion_reglas(semilla=42)

    mejor = resultado["mejor"]
    prueba = resultado["resultado_prueba"]
    historial = resultado["historial"]
    resumen_splits = resultado["resumen_splits"]
    total_prueba = len(resultado["splits"]["prueba"])

    print()
    print("Reparticion de los datos (70/30 estratificado)")
    print("-" * 70)
    print(resumen_splits.to_string(index=False))

    print()
    print("Mejor base de reglas encontrada")
    print("-" * 70)
    print(f"Reglas activas |S|:             {mejor.cantidad_reglas} / {total_prueba}")
    print(f"Balanced Accuracy entrenamiento: {mejor.balanced_accuracy:.4f}")
    print(f"Compacidad C(S) entrenamiento:   {mejor.compacidad:.4f}")
    print(f"Fitness entrenamiento:           {mejor.fitness:.4f}")
    print(f"Generaciones ejecutadas:         {len(historial) - 1}")

    print()
    print("Desempeno en prueba (split 30%)")
    print("-" * 70)
    print(f"Balanced Accuracy: {prueba.balanced_accuracy:.4f}")
    print(f"Compacidad C(S):   {prueba.compacidad:.4f}")
    print(f"Fitness:           {prueba.fitness:.4f}")

    print()
    print("Historial de fitness por generacion")
    print("-" * 70)
    print(historial.to_string(index=False))


if __name__ == "__main__":
    principal()
