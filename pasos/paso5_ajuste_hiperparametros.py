"""PASO 5 · Ajuste de hiperparámetros con Grid Search  (Parte 2 — surprise)

Retos: comparar SVD y KNN en RMSE y obtener el RMSE más bajo posible.

Protocolo sin fuga de información:
    1. GridSearchCV (3 folds) SOLO sobre train (80 %).
    2. Reentrenar la mejor configuración con todo train.
    3. Medir el RMSE final en test (20 %), que ningún modelo vio durante el ajuste.

Genera results/mejores_parametros.json, que usa el paso 6.

Ejecución:  python pasos\\paso5_ajuste_hiperparametros.py [--mostrar]
Duración aproximada: 3-6 minutos (usa todos los núcleos del procesador).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from surprise import SVD, KNNBaseline, KNNWithMeans, accuracy  # noqa: E402
from surprise.model_selection import GridSearchCV, KFold  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import DIR_FIGURAS, DIR_PROCESADO, DIR_TABLAS, RAIZ, SEMILLA  # noqa: E402
from src.estilo import AZUL, GRIS_TEXTO, NARANJA, PALETA, aplicar_estilo, guardar, plt  # noqa: E402
from src.recomendacion import a_dataset_surprise  # noqa: E402

ARCHIVO_PARAMETROS = RAIZ / "results" / "mejores_parametros.json"


def main(mostrar: bool = False) -> None:
    aplicar_estilo(mostrar)
    titulo("Paso 5 · Ajuste de hiperparámetros con Grid Search")
    train = pd.read_csv(DIR_PROCESADO / "train.csv")
    test = pd.read_csv(DIR_PROCESADO / "test.csv")
    data_train = a_dataset_surprise(train)
    cv3 = KFold(n_splits=3, shuffle=True, random_state=SEMILLA)

    # 1. SVD -----------------------------------------------------------------
    seccion("1. Grid search para SVD (2 x 4 x 2 x 3 = 48 combinaciones x 3 folds)")
    nota("""
    - n_factors: número de factores latentes (capacidad del modelo).
    - n_epochs: pasadas del descenso de gradiente.
    - lr_all: tasa de aprendizaje.
    - reg_all: regularización L2 (evita el sobreajuste).
    """)
    grid_svd = {
        "n_factors": [50, 100],
        "n_epochs": [20, 30, 40, 50],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.05, 0.1, 0.15],
        "random_state": [SEMILLA],
    }
    print("\nEntrenando... (1-2 minutos)")
    t0 = time.time()
    gs_svd = GridSearchCV(SVD, grid_svd, measures=["rmse", "mae"], cv=cv3, n_jobs=-1)
    gs_svd.fit(data_train)
    params_svd = dict(gs_svd.best_params["rmse"])
    print(f"Tiempo: {time.time() - t0:.0f} s")
    print(f"Mejor RMSE (CV): {gs_svd.best_score['rmse']:.4f}")
    print(f"Mejores parámetros: {params_svd}")

    res_svd = pd.DataFrame(gs_svd.cv_results)
    cols = ["param_n_factors", "param_n_epochs", "param_lr_all", "param_reg_all", "mean_test_rmse", "mean_test_mae"]
    res_svd = res_svd[cols].sort_values("mean_test_rmse").reset_index(drop=True)
    res_svd.to_csv(DIR_TABLAS / "05_gridsearch_svd.csv", index=False)
    vista = res_svd.head(10).copy()
    vista.columns = ["n_factors", "n_epochs", "lr_all", "reg_all", "RMSE", "MAE"]
    vista.index += 1
    tabla(vista, "Las 10 mejores combinaciones de SVD:", formatters={"RMSE": "{:.4f}".format, "MAE": "{:.4f}".format})

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lr in zip(ejes, [0.005, 0.01]):
        sub = res_svd[(res_svd["param_lr_all"] == lr) & (res_svd["param_n_factors"] == 100)]
        for j, reg in enumerate([0.05, 0.1, 0.15]):
            s = sub[sub["param_reg_all"] == reg].sort_values("param_n_epochs")
            ax.plot(s["param_n_epochs"], s["mean_test_rmse"], marker="o", markersize=5, linewidth=2,
                    color=PALETA[j], label=f"reg_all = {reg}")
        ax.set_title(f"SVD (n_factors=100), lr_all = {lr}")
        ax.set_xlabel("n_epochs")
        ax.set_xticks([20, 30, 40, 50])
    ejes[0].set_ylabel("RMSE (CV 3-fold)")
    ejes[1].legend(title="Regularización")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "05_svd_epochs_regularizacion.png")
    nota(f"""
    Tensión sesgo-varianza: con poca regularización (reg_all = 0.05) más épocas EMPEORAN el RMSE (el modelo
    memoriza el train: sobreajuste). Con más regularización, más épocas sí ayudan. La mejor configuración
    (n_factors={params_svd['n_factors']}, n_epochs={params_svd['n_epochs']}, lr_all={params_svd['lr_all']}, reg_all={params_svd['reg_all']})
    equilibra ambos; 50 factores rinden igual o mejor que 100.
    """)

    # 2. KNN -----------------------------------------------------------------
    seccion("2. Grid search para KNN (2 algoritmos x 4 k x 3 similitudes x 2 enfoques = 48 combinaciones x 3 folds)")
    nota("""
    - k: número de vecinos.
    - similitud: msd, cosine o pearson_baseline.
    - enfoque: usuario-usuario o ítem-ítem.
    - algoritmo: KNNWithMeans (centra en la media) o KNNBaseline (centra en los sesgos b_u, b_i).
    """)
    grid_knn = {
        "k": [20, 40, 60, 80],
        "sim_options": {"name": ["msd", "cosine", "pearson_baseline"], "user_based": [True, False]},
        "verbose": [False],
    }
    print("\nEntrenando... (2-4 minutos)")
    resultados, mejores_knn = [], {}
    t0 = time.time()
    for algo in [KNNWithMeans, KNNBaseline]:
        gs = GridSearchCV(algo, grid_knn, measures=["rmse", "mae"], cv=cv3, n_jobs=-1)
        gs.fit(data_train)
        mejores_knn[algo.__name__] = (gs.best_score["rmse"], gs.best_params["rmse"])
        df = pd.DataFrame(gs.cv_results)
        df["algoritmo"] = algo.__name__
        df["similitud"] = df["param_sim_options"].map(lambda d: d["name"])
        df["enfoque"] = df["param_sim_options"].map(lambda d: "usuario-usuario" if d["user_based"] else "ítem-ítem")
        resultados.append(df[["algoritmo", "enfoque", "similitud", "param_k", "mean_test_rmse", "mean_test_mae"]])
    print(f"Tiempo: {time.time() - t0:.0f} s")

    res_knn = pd.concat(resultados).sort_values("mean_test_rmse").reset_index(drop=True)
    res_knn.to_csv(DIR_TABLAS / "05_gridsearch_knn.csv", index=False)
    for nombre, (score, params) in mejores_knn.items():
        print(f"  {nombre:<13} mejor RMSE (CV) = {score:.4f} | {params}")
    vista = res_knn.head(10).rename(columns={"param_k": "k", "mean_test_rmse": "RMSE", "mean_test_mae": "MAE"})
    vista.index += 1
    tabla(vista, "Las 10 mejores combinaciones de KNN:", formatters={"RMSE": "{:.4f}".format, "MAE": "{:.4f}".format})

    mejor_algo_knn = res_knn.loc[0, "algoritmo"]
    sub = res_knn[res_knn["algoritmo"] == mejor_algo_knn]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, enfoque in zip(ejes, ["usuario-usuario", "ítem-ítem"]):
        for j, sim in enumerate(["msd", "cosine", "pearson_baseline"]):
            s = sub[(sub["enfoque"] == enfoque) & (sub["similitud"] == sim)].sort_values("param_k")
            ax.plot(s["param_k"], s["mean_test_rmse"], marker="o", markersize=5, linewidth=2, color=PALETA[j], label=sim)
        ax.set_title(f"{mejor_algo_knn} — {enfoque}")
        ax.set_xlabel("k (número de vecinos)")
        ax.set_xticks([20, 40, 60, 80])
    ejes[0].set_ylabel("RMSE (CV 3-fold)")
    ejes[1].legend(title="Similitud")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "05_knn_k_similitud.png")
    nota("""
    - KNNBaseline supera a KNNWithMeans: centrar con los sesgos aprendidos b_u + b_i es mejor que centrar solo con la media.
    - Las mejores configuraciones están prácticamente empatadas (usuario-usuario con msd e ítem-ítem con pearson_baseline).
    - El valor de k casi no importa a partir de 40 vecinos: lo que más pesa es el algoritmo y la similitud.
    """)

    # 3. Evaluación final en test -------------------------------------------
    seccion("3. Evaluación final en test (20 % nunca visto)")
    trainset = data_train.build_full_trainset()
    testset = list(test[["user_id", "item_id", "rating"]].itertuples(index=False, name=None))
    params_knn = mejores_knn[mejor_algo_knn][1]
    clase_knn = {"KNNWithMeans": KNNWithMeans, "KNNBaseline": KNNBaseline}[mejor_algo_knn]
    candidatos = {
        "SVD por defecto": SVD(random_state=SEMILLA),
        "SVD ajustado": SVD(**params_svd),
        "KNNWithMeans ítem-ítem por defecto": KNNWithMeans(sim_options={"user_based": False}, verbose=False),
        f"{mejor_algo_knn} ajustado": clase_knn(**params_knn),
    }
    filas = []
    for nombre, algo in candidatos.items():
        algo.fit(trainset)
        pred = algo.test(testset)
        filas.append({"modelo": nombre, "RMSE_test": accuracy.rmse(pred, verbose=False),
                      "MAE_test": accuracy.mae(pred, verbose=False)})
    final = pd.DataFrame(filas).sort_values("RMSE_test").reset_index(drop=True)
    final.index += 1
    final.to_csv(DIR_TABLAS / "05_comparacion_final_test.csv", index_label="puesto")
    tabla(final, ">>> RETO: SVD vs. KNN en test (menor RMSE es mejor)", float_format="{:.4f}".format)

    orden = final.sort_values("RMSE_test", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for y, (n, v) in enumerate(zip(orden["modelo"], orden["RMSE_test"])):
        ax.barh(y, v, color=NARANJA if n.startswith("SVD") else AZUL, alpha=1.0 if "ajustado" in n else 0.45, height=0.6)
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=9, color=GRIS_TEXTO)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels(orden["modelo"])
    ax.set_xlim(0.88, orden["RMSE_test"].max() + 0.02)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE en test (menor es mejor) — eje desde 0.88")
    ax.set_title("SVD vs. KNN: antes y después del ajuste")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "05_svd_vs_knn_test.png")

    rt = final.set_index("modelo")["RMSE_test"]
    config = {
        "mejor_modelo_test": final.iloc[0].modelo,
        "svd": {"params": params_svd, "rmse_cv": gs_svd.best_score["rmse"], "rmse_test": float(rt["SVD ajustado"])},
        "knn": {"algoritmo": mejor_algo_knn, "params": params_knn, "rmse_cv": mejores_knn[mejor_algo_knn][0],
                "rmse_test": float(rt[f"{mejor_algo_knn} ajustado"])},
    }
    ARCHIVO_PARAMETROS.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParámetros ganadores guardados en results\\{ARCHIVO_PARAMETROS.name}")

    titulo("Conclusión del paso 5")
    mejora_svd = rt["SVD por defecto"] - rt["SVD ajustado"]
    mejora_knn = rt["KNNWithMeans ítem-ítem por defecto"] - rt[f"{mejor_algo_knn} ajustado"]
    nota(f"""
    - Respuesta al reto: por defecto KNN y SVD estaban empatados, pero tras el ajuste gana {final.iloc[0].modelo} ({rt['SVD ajustado']:.3f} vs. {rt[f'{mejor_algo_knn} ajustado']:.3f} en test). El RMSE más bajo obtenido es {final.iloc[0].RMSE_test:.4f}.
    - El ajuste mejora SVD en {mejora_svd:.3f} estrellas de RMSE; a KNN le aporta menos ({mejora_knn:.3f}).
    - El RMSE en test es menor que en la validación cruzada porque en CV cada modelo entrena con 2/3 del train (≈53 000 notas) y el modelo final con todo el train (80 000).
    - Trabajo futuro: lr_all quedó en el borde de la grilla (0.01); se podría probar 0.015-0.02 o SVD++.

    Siguiente: python pasos\\paso6_recomendaciones_top5.py
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 5: ajuste de hiperparámetros con GridSearchCV")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    args = parser.parse_args()
    with registrar_salida("paso5_ajuste_hiperparametros"):
        main(args.mostrar)


if __name__ == "__main__":
    cli()
