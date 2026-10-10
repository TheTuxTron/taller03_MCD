"""PASO 6 · Top-5 recomendaciones para un usuario  (Parte 2 — surprise)

Reto: generar las 5 mejores películas para un usuario específico.

Cómo se genera el Top-N:
    1. Se entrena el modelo ajustado (paso 5) con TODAS las calificaciones.
    2. Se predice la nota de cada película que el usuario NO ha visto.
    3. Se ordenan las predicciones y se toman las 5 mayores.

Ejecución:
    python pasos\\paso6_recomendaciones_top5.py                 (usuario 196 por defecto)
    python pasos\\paso6_recomendaciones_top5.py --usuario 13    (cualquier id entre 1 y 943)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from surprise import SVD, KNNBaseline, KNNWithMeans  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import DIR_TABLAS, RAIZ, cargar_procesados  # noqa: E402
from src.recomendacion import a_dataset_surprise, top_n_para_usuario  # noqa: E402

ARCHIVO_PARAMETROS = RAIZ / "results" / "mejores_parametros.json"
MIN_CALIF = 20


def main(usuario: int = 196, min_calif: int = MIN_CALIF) -> None:
    titulo(f"Paso 6 · Top-5 recomendaciones para el usuario {usuario}")
    ratings, peliculas = cargar_procesados()
    if usuario not in set(ratings["user_id"]):
        raise SystemExit(f"El usuario {usuario} no existe. Usa un id entre 1 y {ratings['user_id'].max()}.")
    if not ARCHIVO_PARAMETROS.exists():
        raise SystemExit("No existe results\\mejores_parametros.json: ejecuta antes el paso 5.")
    config = json.loads(ARCHIVO_PARAMETROS.read_text(encoding="utf-8"))
    n_calif = ratings.groupby("item_id").size().rename("num_calificaciones")

    # 1. Perfil del usuario --------------------------------------------------
    seccion("1. ¿Quién es el usuario? (su historial)")
    hist = ratings[ratings["user_id"] == usuario].merge(peliculas, on="item_id")
    print(f"Usuario {usuario}: {len(hist)} películas calificadas, nota media {hist['rating'].mean():.2f}")
    favoritas = hist[hist["rating"] == 5].sort_values("titulo")[["titulo"]].reset_index(drop=True)
    favoritas.index += 1
    tabla(favoritas.head(15), f"Sus películas favoritas (5 estrellas){' — primeras 15' if len(favoritas) > 15 else ''}:")

    # 2. Entrenamiento con todo el dataset -----------------------------------
    seccion("2. Entrenamiento de los modelos ajustados con todo el dataset")
    trainset = a_dataset_surprise(ratings).build_full_trainset()
    svd = SVD(**config["svd"]["params"])
    svd.fit(trainset)
    clase_knn = {"KNNWithMeans": KNNWithMeans, "KNNBaseline": KNNBaseline}[config["knn"]["algoritmo"]]
    knn = clase_knn(**config["knn"]["params"])
    knn.fit(trainset)
    print(f"SVD {config['svd']['params']}")
    print(f"{config['knn']['algoritmo']} {config['knn']['params']}")
    print(f"Entrenados con {trainset.n_ratings:,} calificaciones")

    # 3. Primer intento sin filtros -----------------------------------------
    seccion("3. Primer intento: Top-5 sin filtros")
    for nombre, modelo in [("SVD", svd), (config["knn"]["algoritmo"], knn)]:
        t = top_n_para_usuario(modelo, ratings, peliculas, usuario, n=5).join(n_calif, on="item_id")
        t.index += 1
        tabla(t, f"{nombre}:", formatters={"rating_estimado": "{:.2f}".format})
    nota(f"""
    PROBLEMA: aparecen películas con muy pocas calificaciones (a veces una sola) y nota estimada cercana a 5.
    Con tan poca evidencia la predicción no es confiable: basta que un vecino haya puesto 5. Es un problema
    clásico de los recomendadores (sesgo de baja evidencia).
    SOLUCIÓN: recomendar solo películas con al menos {min_calif} calificaciones (el mismo umbral que GroupLens
    usó para incluir usuarios). Es una regla de negocio común en sistemas reales.
    """)

    # 4. Top-5 final ---------------------------------------------------------
    seccion(f"4. Top-5 final (películas con >= {min_calif} calificaciones)")
    top_svd = top_n_para_usuario(svd, ratings, peliculas, usuario, n=5, min_calificaciones=min_calif).join(n_calif, on="item_id")
    top_svd.index += 1
    top_svd.to_csv(DIR_TABLAS / f"06_top5_usuario_{usuario}_svd.csv", index_label="puesto")
    tabla(top_svd, f">>> RETO: Top-5 para el usuario {usuario} — SVD ajustado (mejor modelo, RMSE test {config['svd']['rmse_test']:.3f})",
          formatters={"rating_estimado": "{:.2f}".format})

    top_knn = top_n_para_usuario(knn, ratings, peliculas, usuario, n=5, min_calificaciones=min_calif).join(n_calif, on="item_id")
    top_knn.index += 1
    top_knn.to_csv(DIR_TABLAS / f"06_top5_usuario_{usuario}_knn.csv", index_label="puesto")
    tabla(top_knn, f"Para contrastar — {config['knn']['algoritmo']} ajustado (RMSE test {config['knn']['rmse_test']:.3f})",
          formatters={"rating_estimado": "{:.2f}".format})

    comunes = sorted(set(top_svd["titulo"]) & set(top_knn["titulo"]))
    print(f"\nPelículas en común entre ambos Top-5 ({len(comunes)} de 5): {', '.join(comunes) if comunes else 'ninguna'}")

    # 5. ¿Personaliza? -------------------------------------------------------
    seccion("5. ¿Las recomendaciones son realmente personalizadas?")
    otros = list(dict.fromkeys([usuario, 196, 405, 13]))[:3]
    comparacion = {}
    for u in otros:
        t = top_n_para_usuario(svd, ratings, peliculas, u, n=5, min_calificaciones=min_calif)
        comparacion[f"Usuario {u} ({(ratings['user_id'] == u).sum()} vistas)"] = [x[:34] for x in t["titulo"]]
    comp = pd.DataFrame(comparacion, index=range(1, 6))
    comp.to_csv(DIR_TABLAS / "06_top5_comparacion_usuarios.csv", index_label="puesto")
    tabla(comp, "Top-5 (SVD) de tres usuarios con perfiles distintos:")
    nota("""
    Si el modelo recomendara lo mismo a todos, sería un ranking de popularidad disfrazado. Las listas son
    distintas para cada perfil: el modelo sí personaliza. Con usuarios de poco historial tiende hacia películas
    de amplio consenso, que es la forma prudente de recomendar cuando hay poca información.
    """)

    titulo("Conclusión del paso 6")
    nota(f"""
    - Top-5 para el usuario {usuario} (SVD ajustado): {'; '.join(top_svd['titulo'])}.
    - SVD y KNN coinciden en {len(comunes)} de 5 películas: el núcleo de la recomendación es robusto al modelo.
    - Lección práctica: un buen RMSE no garantiza un buen Top-N; hizo falta un umbral de evidencia para que las recomendaciones sean confiables.
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 6: Top-5 recomendaciones para un usuario")
    parser.add_argument("--usuario", type=int, default=196, help="id del usuario (1 a 943). Por defecto: 196")
    parser.add_argument("--min-calif", type=int, default=MIN_CALIF,
                        help=f"mínimo de calificaciones de una película para recomendarla. Por defecto: {MIN_CALIF}")
    args = parser.parse_args()
    with registrar_salida(f"paso6_recomendaciones_usuario_{args.usuario}"):
        main(args.usuario, args.min_calif)


if __name__ == "__main__":
    cli()
