import json

from ..entrenamiento import obtener_resultado_entrenamiento


def principal():
    resultado = obtener_resultado_entrenamiento(forzar_reentrenamiento=True)
    mejor_individuo = resultado["mejor_individuo"]
    historial = resultado["historial"]
    tabla_comparativa = resultado["tabla_comparativa"]

    print("Comparacion base vs optimizado")
    print("-" * 90)
    print(tabla_comparativa.to_string(index=False))
    print()
    print("Mejor fitness final")
    print("-" * 90)
    print(f"Fitness: {mejor_individuo.fitness:.4f}")
    print(f"MacroF1 validacion: {mejor_individuo.macro_f1_validacion:.4f}")
    print(f"Recall alto validacion: {mejor_individuo.recall_alto_validacion:.4f}")
    print()
    print("Mejor cromosoma")
    print("-" * 90)
    print(json.dumps([float(valor) for valor in mejor_individuo.cromosoma.tolist()], indent=2))
    print()
    print("Historial de fitness")
    print("-" * 90)
    print(historial.to_string(index=False))

if __name__ == "__main__":
    principal()
