"""Funciones de análisis de redes sobre el grafo bipartito usuario–película."""
from __future__ import annotations

import itertools

import networkx as nx
import pandas as pd


def nodo_usuario(uid: int) -> str:
    return f"u{uid}"


def nodo_pelicula(iid: int) -> str:
    return f"m{iid}"


def construir_grafo_bipartito(ratings: pd.DataFrame) -> nx.Graph:
    """Usuarios (bipartite=0) conectados a las películas que valoraron (bipartite=1).

    La calificación se guarda como atributo 'rating' de la arista; las métricas
    de centralidad se calculan sin pesos (cuenta la interacción, no la nota).
    """
    G = nx.Graph()
    G.add_nodes_from(
        (nodo_usuario(u) for u in ratings["user_id"].unique()), bipartite=0, tipo="usuario"
    )
    G.add_nodes_from(
        (nodo_pelicula(i) for i in ratings["item_id"].unique()), bipartite=1, tipo="pelicula"
    )
    G.add_edges_from(
        (nodo_usuario(u), nodo_pelicula(i), {"rating": r})
        for u, i, r in ratings[["user_id", "item_id", "rating"]].itertuples(index=False)
    )
    return G


def particiones(G: nx.Graph) -> tuple[list[str], list[str]]:
    usuarios = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    peliculas = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
    return usuarios, peliculas


def id_de_nodo(nodo: str) -> int:
    return int(nodo[1:])


def grafo_similitud_peliculas(
    ratings: pd.DataFrame, n_peliculas: int = 80, k_vecinos: int = 4
) -> nx.Graph:
    """Proyección película–película sobre las N películas más valoradas.

    Peso = lift de co-valoración = |A∩B| · N_usuarios / (|A| · |B|).
    El lift corrige el sesgo de popularidad: mide cuánto más se co-valoran dos
    películas de lo esperado si fueran independientes. Cada película se une a
    sus k vecinos de mayor lift, lo que produce un grafo disperso (apto para
    Girvan–Newman) y conectado.
    """
    n_usuarios = ratings["user_id"].nunique()
    top = list(ratings["item_id"].value_counts().head(n_peliculas).index)
    valoradores = {m: set(ratings.loc[ratings["item_id"] == m, "user_id"]) for m in top}

    lift = {}
    for a, b in itertools.combinations(top, 2):
        valor = len(valoradores[a] & valoradores[b]) * n_usuarios / (
            len(valoradores[a]) * len(valoradores[b])
        )
        lift[(a, b)] = lift[(b, a)] = valor

    P = nx.Graph()
    P.add_nodes_from(top)
    for a in top:
        vecinos = sorted((b for b in top if b != a), key=lambda b: -lift[(a, b)])[:k_vecinos]
        for b in vecinos:
            P.add_edge(a, b, weight=lift[(a, b)])
    return P


def girvan_newman_mejor_particion(G: nx.Graph, max_niveles: int = 40):
    """Recorre la jerarquía de Girvan–Newman y devuelve el nivel de mayor modularidad.

    Retorna (mejor_particion, historial) donde historial es un DataFrame con
    el número de comunidades y la modularidad de cada nivel.
    """
    historial = []
    mejor, mejor_q = None, float("-inf")
    for nivel, comunidades in enumerate(
        itertools.islice(nx.community.girvan_newman(G), max_niveles), start=1
    ):
        comunidades = [set(c) for c in comunidades]
        q = nx.community.modularity(G, comunidades)
        historial.append({"nivel": nivel, "n_comunidades": len(comunidades), "modularidad": q})
        if q > mejor_q:
            mejor, mejor_q = comunidades, q
    mejor = sorted(mejor, key=len, reverse=True)
    return mejor, pd.DataFrame(historial)
