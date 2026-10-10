"""Utilidades para los sistemas de recomendación con la librería surprise."""
from __future__ import annotations

import pandas as pd
from surprise import Dataset, Reader

READER = Reader(rating_scale=(1, 5))


def a_dataset_surprise(ratings: pd.DataFrame) -> Dataset:
    """Convierte un DataFrame (user_id, item_id, rating) al formato de surprise."""
    return Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], READER)


def top_n_para_usuario(
    modelo,
    ratings: pd.DataFrame,
    peliculas: pd.DataFrame,
    user_id: int,
    n: int = 5,
    min_calificaciones: int = 0,
) -> pd.DataFrame:
    """Predice la nota de cada película que el usuario NO ha valorado y devuelve las n mejores.

    El modelo debe estar entrenado (fit) con un trainset que contenga al usuario.
    min_calificaciones descarta películas con muy pocas valoraciones, cuya nota
    estimada es poco confiable (pocas evidencias).
    """
    vistas = set(ratings.loc[ratings["user_id"] == user_id, "item_id"])
    conteo = ratings.groupby("item_id").size()
    candidatas = [
        i for i in peliculas["item_id"]
        if i not in vistas and conteo.get(i, 0) >= min_calificaciones
    ]
    preds = [modelo.predict(user_id, i) for i in candidatas]
    df = pd.DataFrame(
        {
            "item_id": [p.iid for p in preds],
            "rating_estimado": [p.est for p in preds],
            "imposible": [p.details.get("was_impossible", False) for p in preds],
        }
    )
    df = df[~df["imposible"]].drop(columns="imposible")
    df = df.merge(peliculas[["item_id", "titulo"]], on="item_id")
    # Desempate determinista: mayor estimación y, a igualdad, menor id.
    df = df.sort_values(["rating_estimado", "item_id"], ascending=[False, True]).head(n)
    return df.reset_index(drop=True)[["item_id", "titulo", "rating_estimado"]]
