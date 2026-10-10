"""PASO 1 · Preparación del conjunto de datos (MovieLens 100K)

Descarga el dataset, revisa su estructura y genera los archivos que usan los
demás pasos:
    data/processed/ratings.csv     interacciones usuario–película
    data/processed/peliculas.csv   id, título y año
    data/processed/train.csv       80 % para entrenar/ajustar (partición fija)
    data/processed/test.csv        20 % reservado para la evaluación final

Ejecución:  python pasos\\paso1_preparacion_datos.py [--mostrar]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import (  # noqa: E402
    DIR_FIGURAS, DIR_PROCESADO, SEMILLA, asegurar_directorios,
    cargar_peliculas, cargar_ratings, descargar_ml100k,
)
from src.estilo import AZUL, GRIS_TEXTO, aplicar_estilo, guardar, plt  # noqa: E402


def main(mostrar: bool = False) -> None:
    aplicar_estilo(mostrar)
    asegurar_directorios()
    titulo("Paso 1 · Preparación de datos — MovieLens 100K")

    # 1. Descarga ------------------------------------------------------------
    seccion("1. Descarga")
    descargar_ml100k()

    # 2. Carga y estructura --------------------------------------------------
    seccion("2. Carga y estructura")
    ratings = cargar_ratings()
    peliculas = cargar_peliculas()
    print(f"ratings: {ratings.shape[0]:,} filas x {ratings.shape[1]} columnas | películas: {len(peliculas):,}")
    tabla(ratings.head(), "Primeras filas de u.data:")

    n_usuarios = ratings["user_id"].nunique()
    n_peliculas = ratings["item_id"].nunique()
    densidad = len(ratings) / (n_usuarios * n_peliculas)
    resumen = pd.Series({
        "Calificaciones": f"{len(ratings):,}",
        "Usuarios": n_usuarios,
        "Películas": n_peliculas,
        "Densidad de la matriz usuario x película": f"{densidad:.2%}",
        "Calificación promedio": round(ratings["rating"].mean(), 3),
        "Mín. calificaciones por usuario": ratings.groupby("user_id").size().min(),
        "Periodo": f"{ratings['fecha'].min():%Y-%m-%d} a {ratings['fecha'].max():%Y-%m-%d}",
        "Duplicados (usuario, película)": int(ratings.duplicated(["user_id", "item_id"]).sum()),
        "Valores nulos": int(ratings.isna().sum().sum()),
    }, name="valor")
    tabla(resumen.to_frame(), "Resumen del dataset:")
    nota(f"""
    La matriz usuario x película está casi vacía ({densidad:.1%} de celdas con dato). Ese es el problema
    que resuelve un sistema de recomendación: estimar el {1 - densidad:.0%} restante.
    Cada usuario tiene al menos 20 calificaciones (así lo construyó GroupLens): no hay usuarios "fríos".
    """)

    # 3. Distribuciones ------------------------------------------------------
    seccion("3. Distribuciones")
    dist = ratings["rating"].value_counts().sort_index()
    tabla(dist.rename("calificaciones").to_frame().T, "Calificaciones por número de estrellas:")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].bar(dist.index, dist.values, color=AZUL, width=0.6)
    for x, y in zip(dist.index, dist.values):
        ejes[0].text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9, color=GRIS_TEXTO)
    ejes[0].set_title("Distribución de calificaciones")
    ejes[0].set_xlabel("Estrellas")
    ejes[0].set_ylabel("Número de calificaciones")

    por_pelicula = ratings.groupby("item_id").size()
    ejes[1].hist(por_pelicula, bins=50, color=AZUL, edgecolor="white", linewidth=0.5)
    ejes[1].set_title("Calificaciones por película (cola larga)")
    ejes[1].set_xlabel("Número de calificaciones recibidas")
    ejes[1].set_ylabel("Películas")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "01_distribuciones.png")

    nota(f"""
    Las notas se concentran en 3-4 estrellas (sesgo positivo típico).
    La popularidad sigue una cola larga: la película más vista tiene {por_pelicula.max()} calificaciones,
    pero la mediana es de solo {int(por_pelicula.median())}. En el grafo esto se verá como nodos "hub".
    """)

    # 4. Guardado y partición train/test ------------------------------------
    seccion("4. Guardado y partición train/test (80/20, semilla fija)")
    ratings[["user_id", "item_id", "rating", "timestamp"]].to_csv(DIR_PROCESADO / "ratings.csv", index=False)
    peliculas.to_csv(DIR_PROCESADO / "peliculas.csv", index=False)

    test = ratings.sample(frac=0.2, random_state=SEMILLA)
    train = ratings.drop(test.index)
    train[["user_id", "item_id", "rating"]].to_csv(DIR_PROCESADO / "train.csv", index=False)
    test[["user_id", "item_id", "rating"]].to_csv(DIR_PROCESADO / "test.csv", index=False)

    print(f"train: {len(train):,} filas | test: {len(test):,} filas")
    print(f"Filas de test con usuario no visto en train : {(~test['user_id'].isin(train['user_id'])).sum()}")
    print(f"Filas de test con película no vista en train: {(~test['item_id'].isin(train['item_id'])).sum()}")
    nota("""
    La partición se hace una sola vez. El ajuste de hiperparámetros (paso 5) usa solo train y test queda
    reservado para la evaluación final: todos los modelos se comparan sobre exactamente los mismos datos.
    """)

    titulo("Conclusión del paso 1")
    nota(f"""
    Dataset listo: {n_usuarios} usuarios, {n_peliculas:,} películas y {len(ratings):,} calificaciones, sin nulos ni duplicados.

    Siguiente: python pasos\\paso2_red_centralidad_clustering.py
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 1: preparación de MovieLens 100K")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    args = parser.parse_args()
    with registrar_salida("paso1_preparacion_datos"):
        main(args.mostrar)


if __name__ == "__main__":
    cli()
