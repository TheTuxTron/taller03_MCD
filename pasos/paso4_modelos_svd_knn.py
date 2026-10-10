"""PASO 4 · Modelos de recomendación: SVD vs. KNN con parámetros por defecto  (Parte 2 — surprise)

Filtrado colaborativo: predecir la nota r_ui que el usuario u daría a la
película i usando solo la matriz de calificaciones.

    SVD                 cada usuario y película = k factores latentes;  r_ui = mu + b_u + b_i + q_i · p_u
    KNN usuario–usuario promedio de las notas que dieron a i los k usuarios más parecidos a u
    KNN ítem–ítem       promedio de las notas que u dio a las k películas más parecidas a i

Métricas: RMSE (penaliza más los errores grandes) y MAE, ambas en estrellas.
Validación cruzada de 5 particiones sobre train (80 %).

Ejecución:  python pasos\\paso4_modelos_svd_knn.py [--mostrar]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from surprise import SVD, BaselineOnly, KNNBasic, KNNWithMeans, NormalPredictor  # noqa: E402
from surprise.model_selection import KFold, cross_validate  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import DIR_FIGURAS, DIR_PROCESADO, DIR_TABLAS, SEMILLA  # noqa: E402
from src.estilo import AZUL, GRIS_SUAVE, GRIS_TEXTO, NARANJA, aplicar_estilo, guardar, plt  # noqa: E402
from src.recomendacion import a_dataset_surprise  # noqa: E402


def main(mostrar: bool = False) -> None:
    aplicar_estilo(mostrar)
    titulo("Paso 4 · SVD vs. KNN con parámetros por defecto")

    seccion("1. Datos en formato surprise")
    train = pd.read_csv(DIR_PROCESADO / "train.csv")
    data_train = a_dataset_surprise(train)
    print(f"Calificaciones de entrenamiento: {len(train):,} (escala 1-5)")

    seccion("2. Validación cruzada (5 folds) de los modelos base")
    nota("""
    Además de SVD y KNN se incluyen dos referencias:
    - NormalPredictor: predice al azar según la distribución de notas (piso de comparación).
    - BaselineOnly: solo sesgos mu + b_u + b_i ("este usuario es exigente, esta película gusta a todos").
    KNNWithMeans centra las notas en la media de cada usuario/ítem antes de promediar vecinos.
    """)
    print()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    modelos = {
        "NormalPredictor (aleatorio)": NormalPredictor(),
        "BaselineOnly (solo sesgos)": BaselineOnly(verbose=False),
        "SVD": SVD(random_state=SEMILLA),
        "KNNBasic usuario-usuario": KNNBasic(sim_options={"user_based": True}, verbose=False),
        "KNNBasic ítem-ítem": KNNBasic(sim_options={"user_based": False}, verbose=False),
        "KNNWithMeans usuario-usuario": KNNWithMeans(sim_options={"user_based": True}, verbose=False),
        "KNNWithMeans ítem-ítem": KNNWithMeans(sim_options={"user_based": False}, verbose=False),
    }
    filas = []
    for nombre, algo in modelos.items():
        t0 = time.time()
        res = cross_validate(algo, data_train, measures=["RMSE", "MAE"], cv=kf, verbose=False)
        filas.append({"modelo": nombre, "RMSE": res["test_rmse"].mean(), "RMSE_std": res["test_rmse"].std(),
                      "MAE": res["test_mae"].mean(), "tiempo_s": time.time() - t0})
        print(f"  {nombre:<30} RMSE = {filas[-1]['RMSE']:.4f}   ({filas[-1]['tiempo_s']:.1f} s)")

    base = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    base.index += 1
    base.to_csv(DIR_TABLAS / "04_modelos_base_cv.csv", index_label="puesto")
    tabla(base, "Ranking (menor RMSE es mejor):",
          formatters={"RMSE": "{:.4f}".format, "RMSE_std": "{:.4f}".format, "MAE": "{:.4f}".format, "tiempo_s": "{:.1f}".format})

    orden = base.sort_values("RMSE", ascending=False)
    color = [NARANJA if n == "SVD" else (GRIS_SUAVE if "Normal" in n or "Baseline" in n else AZUL) for n in orden["modelo"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(orden["modelo"], orden["RMSE"], color=color, height=0.6)
    for y, v in enumerate(orden["RMSE"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=GRIS_TEXTO)
    ax.set_xlim(0.85, orden["RMSE"].max() + 0.06)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE (validación cruzada 5-fold, menor es mejor) — eje desde 0.85")
    ax.set_title("Modelos con parámetros por defecto")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "04_rmse_modelos_base.png")

    r = base.set_index("modelo")["RMSE"]
    nota(f"""
    - Todos los modelos de filtrado colaborativo reducen el error a menos de la mitad que el azar (≈0.94-0.99 frente a {r['NormalPredictor (aleatorio)']:.2f}).
    - Con parámetros por defecto el mejor es {base.iloc[0].modelo} (RMSE {base.iloc[0].RMSE:.3f}), seguido muy de cerca por SVD ({r['SVD']:.3f}): prácticamente un empate.
    - Centrar las notas (KNNWithMeans) mejora mucho a KNNBasic ({r['KNNWithMeans ítem-ítem']:.3f} vs. {r['KNNBasic ítem-ítem']:.3f}): corrige que hay usuarios que ponen 5 a todo y otros que nunca pasan de 3.
    - Ítem-ítem supera a usuario-usuario: las películas tienen perfiles de valoración más estables que los usuarios.
    - BaselineOnly, que solo usa sesgos, queda en {r['BaselineOnly (solo sesgos)']:.3f}, casi igual que SVD por defecto. Gran parte de la señal está en "qué tan generoso es el usuario" y "qué tan buena es la película"; para que SVD aporte la parte personalizada hay que ajustarlo (paso 5).
    """)

    titulo("Conclusión del paso 4")
    nota("""
    Por defecto KNN ítem-ítem y SVD están empatados. El paso 5 ajusta los hiperparámetros de ambos.

    Siguiente: python pasos\\paso5_ajuste_hiperparametros.py
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 4: SVD vs. KNN con parámetros por defecto")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    args = parser.parse_args()
    with registrar_salida("paso4_modelos_svd_knn"):
        main(args.mostrar)


if __name__ == "__main__":
    cli()
