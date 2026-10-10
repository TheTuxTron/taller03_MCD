"""Utilidades para mostrar resultados en la terminal de forma legible.

Todo lo que se imprime se guarda además en results/salidas/<paso>.txt, para
poder revisarlo después (o mostrarlo si algo falla durante la exposición).
"""
from __future__ import annotations

import sys
import textwrap
from contextlib import contextmanager

import pandas as pd

from .datos import DIR_SALIDAS, RAIZ

ANCHO = 100

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 90)


def _forzar_utf8() -> None:
    """La consola de Windows no siempre usa UTF-8: evita errores con tildes y símbolos."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


class _Duplicador:
    """Escribe a la vez en la consola y en un archivo."""

    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)
        return len(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


def ruta_relativa(ruta) -> str:
    try:
        return str(ruta.relative_to(RAIZ))
    except ValueError:
        return str(ruta)


@contextmanager
def registrar_salida(nombre: str):
    """Duplica la salida de la consola en results/salidas/<nombre>.txt."""
    _forzar_utf8()
    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    archivo = DIR_SALIDAS / f"{nombre}.txt"
    original = sys.stdout
    with open(archivo, "w", encoding="utf-8") as f:
        sys.stdout = _Duplicador(original, f)
        try:
            yield
        finally:
            sys.stdout = original
    print(f"\n(Salida completa guardada en {ruta_relativa(archivo)})")


def titulo(texto: str) -> None:
    print("\n" + "=" * ANCHO)
    print(texto.upper())
    print("=" * ANCHO)


def seccion(texto: str) -> None:
    print("\n" + texto)
    print("-" * len(texto))


def tabla(df: pd.DataFrame, encabezado: str | None = None, **kwargs) -> None:
    if encabezado:
        print(f"\n{encabezado}")
    print(df.to_string(**kwargs))


def nota(texto: str) -> None:
    """Imprime una interpretación con sangría.

    Las líneas que empiezan con '- ' son viñetas; las demás líneas consecutivas
    se unen en un mismo párrafo y se reajustan al ancho de la consola.
    """
    bloques: list[tuple[str, str]] = []
    for linea in textwrap.dedent(texto).strip().splitlines():
        linea = linea.strip()
        if not linea:
            bloques.append(("vacio", ""))
        elif linea.startswith("- "):
            bloques.append(("viñeta", linea[2:]))
        elif bloques and bloques[-1][0] == "parrafo":
            bloques[-1] = ("parrafo", bloques[-1][1] + " " + linea)
        else:
            bloques.append(("parrafo", linea))
    print()
    for tipo, contenido in bloques:
        if tipo == "vacio":
            print()
        elif tipo == "viñeta":
            print(textwrap.fill(contenido, ANCHO, initial_indent="   • ", subsequent_indent="     "))
        else:
            print(textwrap.fill(contenido, ANCHO, initial_indent="  » ", subsequent_indent="    "))
