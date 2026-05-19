"""Funciones simples para guardar resultados tabulares y estructurados."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def guardar_json(ruta: Path, contenido: dict):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_csv(ruta: Path, filas: list[dict]):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def limpiar_pngs_raiz(carpeta: Path):
    if not carpeta.exists():
        return
    for archivo in carpeta.glob("*.png"):
        archivo.unlink()

