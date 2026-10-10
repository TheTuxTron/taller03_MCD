"""Carga y preparación del dataset MovieLens 100K (ml-100k).

Todas las rutas se resuelven a partir de la raíz del proyecto, así que los
scripts funcionan sin importar desde qué carpeta se ejecuten.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "ml-100k"
DIR_PROCESADO = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "results" / "figures"
DIR_TABLAS = RAIZ / "results" / "tables"
DIR_SALIDAS = RAIZ / "results" / "salidas"

URL_ML100K = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SEMILLA = 42


def asegurar_directorios() -> None:
    for d in (DIR_RAW, DIR_PROCESADO, DIR_FIGURAS, DIR_TABLAS, DIR_SALIDAS):
        d.mkdir(parents=True, exist_ok=True)


def _usar_certificados_del_sistema() -> None:
    """Hace que Python valide HTTPS con el almacén de certificados de Windows.

    El Python de python.org trae su propia lista de certificados y no ve los que
    Windows tiene instalados (por ejemplo, el de un antivirus o proxy corporativo
    que inspecciona HTTPS). Eso provoca el error CERTIFICATE_VERIFY_FAILED.
    truststore resuelve el problema sin desactivar la verificación.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def descargar_ml100k(forzar: bool = False) -> Path:
    """Descarga el zip oficial de GroupLens y lo extrae en data/ml-100k/."""
    asegurar_directorios()
    if (DIR_RAW / "u.data").exists() and not forzar:
        print(f"Dataset ya disponible en {DIR_RAW.relative_to(RAIZ)}")
        return DIR_RAW
    print(f"Descargando {URL_ML100K} ...")
    _usar_certificados_del_sistema()
    try:
        with urlopen(URL_ML100K, timeout=60) as resp:
            contenido = resp.read()
    except URLError as e:
        raise SystemExit(
            f"\nNo se pudo descargar el dataset: {e.reason}\n\n"
            "Solución manual:\n"
            "  1. Descarga ml-100k.zip desde https://grouplens.org/datasets/movielens/100k/\n"
            f"  2. Copia los archivos u.data y u.item del zip en: {DIR_RAW}\n"
            "  3. Vuelve a ejecutar este paso.\n"
        )
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        for nombre in z.namelist():
            if nombre.endswith("/"):
                continue
            destino = DIR_RAW / Path(nombre).name
            destino.write_bytes(z.read(nombre))
    print(f"Listo: archivos extraídos en {DIR_RAW.relative_to(RAIZ)}")
    return DIR_RAW


def cargar_ratings() -> pd.DataFrame:
    """u.data: user_id, item_id, rating, timestamp (separado por tabulador)."""
    df = pd.read_csv(
        DIR_RAW / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    df["fecha"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def cargar_peliculas() -> pd.DataFrame:
    """u.item: solo id y título (el título incluye el año de estreno)."""
    df = pd.read_csv(
        DIR_RAW / "u.item",
        sep="|",
        header=None,
        usecols=[0, 1],
        names=["item_id", "titulo"],
        encoding="latin-1",
    )
    df["anio"] = df["titulo"].str.extract(r"\((\d{4})\)\s*$").astype("float")
    return df


def cargar_procesados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los archivos generados por el paso 1."""
    ratings = pd.read_csv(DIR_PROCESADO / "ratings.csv")
    peliculas = pd.read_csv(DIR_PROCESADO / "peliculas.csv")
    return ratings, peliculas
