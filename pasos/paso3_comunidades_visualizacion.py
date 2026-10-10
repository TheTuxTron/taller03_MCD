"""PASO 3 · Detección de comunidades y visualización  (Parte 1 — networkx)

Reto: visualizar la red con enfoque en las comunidades.

Problema de escala de Girvan–Newman: elimina la arista de mayor intermediación
y RECALCULA la intermediación tras cada eliminación (O(|V||E|) por paso). Sobre
el grafo completo (2 625 nodos, 100 000 aristas) es inviable. Estrategia:
    1. Girvan–Newman sobre la red de similitud de las 80 películas más populares.
    2. Louvain (casi lineal) sobre el grafo bipartito completo, como contraste.
    3. Comparar ambas particiones y visualizarlas.

Ejecución:  python pasos\\paso3_comunidades_visualizacion.py [--mostrar]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import DIR_FIGURAS, DIR_TABLAS, SEMILLA, cargar_procesados  # noqa: E402
from src.estilo import AZUL, GRIS_SUAVE, GRIS_TEXTO, PALETA, aplicar_estilo, guardar, plt  # noqa: E402
from src.red import (  # noqa: E402
    construir_grafo_bipartito, girvan_newman_mejor_particion, grafo_similitud_peliculas, id_de_nodo,
)


def main(mostrar: bool = False) -> None:
    aplicar_estilo(mostrar)
    titulo("Paso 3 · Detección de comunidades y visualización")
    ratings, peliculas = cargar_procesados()
    titulo_de = dict(zip(peliculas["item_id"], peliculas["titulo"]))
    anio = dict(zip(peliculas["item_id"], peliculas["anio"]))
    n_calif = ratings.groupby("item_id").size()

    def corto(item_id, n=28):
        return titulo_de[item_id].rsplit(" (", 1)[0][:n]

    # 1. Red de similitud ---------------------------------------------------
    seccion("1. Red de similitud entre las 80 películas más populares")
    nota("""
    Peso entre dos películas = lift de co-valoración = |Ua ∩ Ub| · |U| / (|Ua| · |Ub|), donde Ua son los usuarios
    que valoraron a. Un lift de 1.5 significa que se co-valoran un 50 % más de lo esperado por azar. Contar
    co-valoraciones brutas haría que todo se conecte con los blockbusters; el lift corrige ese sesgo.
    Cada película se une a sus 4 vecinas de mayor lift: grafo disperso y conexo.
    """)
    P = grafo_similitud_peliculas(ratings, n_peliculas=80, k_vecinos=4)
    print(f"\nNodos: {P.number_of_nodes()} | Aristas: {P.number_of_edges()} | Conexo: {nx.is_connected(P)}")

    # 2. Girvan–Newman ------------------------------------------------------
    seccion("2. Girvan–Newman")
    comunidades_gn, historial = girvan_newman_mejor_particion(P, max_niveles=40)
    mejor = historial.loc[historial["modularidad"].idxmax()]
    tabla(historial.head(10).set_index("nivel").round(3), "Modularidad Q en los primeros niveles de la jerarquía:")
    print(f"\nMejor corte: {int(mejor['n_comunidades'])} comunidades, modularidad Q = {mejor['modularidad']:.3f}")
    nota("Modularidad: fracción de aristas dentro de las comunidades menos la esperada al azar. Sobre ~0.3 ya indica estructura clara.")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(historial["n_comunidades"], historial["modularidad"], color=AZUL, linewidth=2, marker="o", markersize=4)
    ax.scatter([mejor["n_comunidades"]], [mejor["modularidad"]], s=90, color=AZUL, edgecolors="white", linewidths=2, zorder=3)
    ax.annotate(f"máximo: {int(mejor['n_comunidades'])} comunidades\nQ = {mejor['modularidad']:.3f}",
                (mejor["n_comunidades"], mejor["modularidad"]), xytext=(12, -32), textcoords="offset points",
                color=GRIS_TEXTO, fontsize=9)
    ax.set_title("Girvan–Newman: modularidad según el número de comunidades")
    ax.set_xlabel("Número de comunidades")
    ax.set_ylabel("Modularidad Q")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "03_girvan_newman_modularidad.png")

    filas = []
    for k, com in enumerate(comunidades_gn, start=1):
        ids = sorted(com, key=lambda i: -n_calif[i])
        filas.append({
            "comunidad": k,
            "n_peliculas": len(com),
            "anio_mediano": int(np.median([anio[i] for i in com])),
            "%_estrenos_96_97": f"{np.mean([anio[i] >= 1996 for i in com]):.0%}",
            "peliculas_representativas": "; ".join(corto(i, 30) for i in ids[:5]),
        })
    tabla_gn = pd.DataFrame(filas).set_index("comunidad")
    tabla_gn.to_csv(DIR_TABLAS / "comunidades_girvan_newman.csv")
    tabla(tabla_gn, "Comunidades encontradas por Girvan–Newman:")
    nota("""
    Las comunidades no son géneros puros sino PÚBLICOS (grupos de películas que ve el mismo tipo de usuario):
    - 1 · Culto y saga Star Wars: Star Wars, Return of the Jedi, Raiders, Pulp Fiction, Silence of the Lambs.
    - 2 · Taquilla 1995-96: Fargo, Toy Story, Independence Day, Twelve Monkeys.
    - 3 · Cartelera 1997: Contact, Liar Liar, Scream, Air Force One, Titanic (100 % estrenos 1996-97).
    - 4 · Populares familiares de los 80-90: Back to the Future, Forrest Gump, E.T., Monty Python.
    - 5 · Clásicos del cine: The Godfather, Casablanca, 2001, Wizard of Oz.
    La comunidad 3 es la más aislada (se ve separada en el gráfico). La ÉPOCA (cuándo se estrenó la película y
    cuándo llegó el usuario) organiza la red tanto como el género.
    """)

    # 3. Visualización de la red de películas -------------------------------
    seccion("3. Visualización de la red de películas por comunidad")
    color_com = {n: PALETA[k] for k, com in enumerate(comunidades_gn) for n in com}
    pos = nx.spring_layout(P, weight="weight", seed=SEMILLA, k=0.35, iterations=200)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_axis_off()
    nx.draw_networkx_edges(P, pos, ax=ax, edge_color=GRIS_SUAVE, width=0.7, alpha=0.8)
    nx.draw_networkx_nodes(P, pos, ax=ax, node_size=[60 + n_calif[n] * 0.9 for n in P.nodes],
                           node_color=[color_com[n] for n in P.nodes], edgecolors="white", linewidths=1.5)
    for com in comunidades_gn:  # etiquetas: las 2 películas más valoradas de cada comunidad
        for n in sorted(com, key=lambda i: -n_calif[i])[:2]:
            x, y = pos[n]
            ax.text(x, y + 0.045, corto(n), fontsize=8, ha="center", color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    leyenda = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETA[k], markersize=10,
                      label=f"Comunidad {k + 1} ({len(c)} películas)") for k, c in enumerate(comunidades_gn)]
    ax.legend(handles=leyenda, loc="upper right", fontsize=9, title="Girvan–Newman", title_fontsize=9)
    ax.set_title("Comunidades de películas (Girvan–Newman) — tamaño del nodo = número de calificaciones", fontsize=12)
    guardar(fig, DIR_FIGURAS / "03_red_peliculas_comunidades.png")

    # 4. Louvain sobre el grafo completo ------------------------------------
    seccion("4. Louvain sobre el grafo bipartito completo")
    G = construir_grafo_bipartito(ratings)
    comunidades_lv = sorted(nx.community.louvain_communities(G, seed=SEMILLA), key=len, reverse=True)
    q_lv = nx.community.modularity(G, comunidades_lv)
    print(f"Louvain: {len(comunidades_lv)} comunidades, modularidad Q = {q_lv:.3f}")

    primera_calif = ratings.groupby("user_id")["timestamp"].min()
    filas = []
    for k, com in enumerate(comunidades_lv, start=1):
        us = [id_de_nodo(n) for n in com if n.startswith("u")]
        ms = sorted((id_de_nodo(n) for n in com if n.startswith("m")), key=lambda i: -n_calif[i])
        filas.append({
            "comunidad": k,
            "usuarios": len(us),
            "peliculas": len(ms),
            "anio_mediano": int(np.nanmedian([anio[i] for i in ms])),
            "%_usuarios_llegados_1998": f"{np.mean(pd.to_datetime(primera_calif[us], unit='s') >= '1998-01-01'):.0%}",
            "peliculas_mas_vistas": "; ".join(corto(i, 26) for i in ms[:4]),
        })
    tabla_lv = pd.DataFrame(filas).set_index("comunidad")
    tabla_lv.to_csv(DIR_TABLAS / "comunidades_louvain.csv")
    tabla(tabla_lv, "Comunidades encontradas por Louvain:")
    nota(f"""
    Modularidad moderada (Q = {q_lv:.2f}): hay comunidades, pero muy solapadas porque casi todos ven los mismos
    éxitos. Reaparece el patrón temporal: la comunidad 4 reúne los estrenos de 1997 y es la de más usuarios
    recién llegados, mientras que la comunidad 2 agrupa los éxitos de los 80.
    """)

    etiqueta_gn = {n: k for k, c in enumerate(comunidades_gn) for n in c}
    etiqueta_lv = {id_de_nodo(n): k for k, c in enumerate(comunidades_lv) for n in c if n.startswith("m")}
    comunes = list(P.nodes)
    ari = adjusted_rand_score([etiqueta_gn[n] for n in comunes], [etiqueta_lv[n] for n in comunes])
    cruce = pd.crosstab(pd.Series([etiqueta_gn[n] + 1 for n in comunes], name="Girvan-Newman"),
                        pd.Series([etiqueta_lv[n] + 1 for n in comunes], name="Louvain"))
    print(f"\nÍndice de Rand ajustado GN vs. Louvain (80 películas): {ari:.3f}")
    tabla(cruce, "Tabla cruzada (filas: comunidad GN, columnas: comunidad Louvain):")
    nota("""
    Acuerdo moderado (0 sería azar, 1 coincidencia total). Las comunidades "Taquilla 1995-96" y "Cartelera 1997"
    de Girvan–Newman coinciden casi exactamente con las comunidades 3 y 4 de Louvain; los tres bloques más
    antiguos Louvain los reparte entre sus comunidades 1 y 2. Los métodos difieren a propósito: GN trabaja sobre
    una red corregida por popularidad con 80 películas; Louvain sobre las 100 000 interacciones crudas.
    """)

    # 5. Visualización del grafo bipartito ----------------------------------
    seccion("5. Visualización del grafo bipartito por comunidades (Louvain)")
    muestra = []
    for com in comunidades_lv:
        muestra += sorted((n for n in com if n.startswith("u")), key=lambda n: -G.degree(n))[:10]
        muestra += sorted((n for n in com if n.startswith("m")), key=lambda n: -G.degree(n))[:10]
    H = G.subgraph(muestra).copy()
    color_lv = {n: PALETA[k] for k, c in enumerate(comunidades_lv) for n in c}

    pos = {}
    K = len(comunidades_lv)
    for k, com in enumerate(comunidades_lv):  # cada comunidad ocupa su propia región del plano
        centro = np.array([np.cos(2 * np.pi * k / K), np.sin(2 * np.pi * k / K)]) * 1.6
        sub_pos = nx.spring_layout(H.subgraph([n for n in H if n in com]), seed=SEMILLA, k=0.6)
        for n, p in sub_pos.items():
            pos[n] = centro + 0.55 * np.asarray(p)

    intra = [(a, b) for a, b in H.edges if color_lv[a] == color_lv[b]]
    inter = [(a, b) for a, b in H.edges if color_lv[a] != color_lv[b]]
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_axis_off()
    nx.draw_networkx_edges(H, pos, edgelist=inter, ax=ax, edge_color=GRIS_SUAVE, width=0.3, alpha=0.25)
    nx.draw_networkx_edges(H, pos, edgelist=intra, ax=ax, width=0.6, alpha=0.45, edge_color=[color_lv[a] for a, b in intra])
    nodos_u = [n for n in H if n.startswith("u")]
    nodos_m = [n for n in H if n.startswith("m")]
    nx.draw_networkx_nodes(H, pos, nodelist=nodos_u, node_shape="o", node_size=70, ax=ax,
                           node_color=[color_lv[n] for n in nodos_u], edgecolors="white", linewidths=1)
    nx.draw_networkx_nodes(H, pos, nodelist=nodos_m, node_shape="s", node_size=150, ax=ax,
                           node_color=[color_lv[n] for n in nodos_m], edgecolors="white", linewidths=1.5)
    for com in comunidades_lv:
        for n in sorted((n for n in com if n.startswith("m")), key=lambda n: -G.degree(n))[:2]:
            x, y = pos[n]
            ax.text(x, y + 0.05, corto(id_de_nodo(n), 26), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    leyenda = [Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETA[k], markersize=10,
                      label=f"Comunidad {k + 1}") for k in range(K)]
    leyenda += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Usuario"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=9, label="Película")]
    ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, title="Louvain", title_fontsize=9)
    ax.set_title(f"Grafo bipartito usuario–película por comunidad (muestra de {H.number_of_nodes()} nodos "
                 f"y {H.number_of_edges()} aristas)", fontsize=12)
    guardar(fig, DIR_FIGURAS / "03_bipartito_comunidades_louvain.png")

    pct_inter = len(inter) / H.number_of_edges()
    print(f"Aristas dentro de la misma comunidad: {len(intra)} | entre comunidades: {len(inter)} ({pct_inter:.0%})")
    nota(f"""
    Muestra: los 10 usuarios más activos y las 10 películas más vistas de cada comunidad (círculos = usuarios,
    cuadrados = películas). Aunque cada comunidad tiene sus aristas internas (en color), el {pct_inter:.0%} de las
    aristas cruza entre comunidades (en gris): los usuarios más activos ven de todo. Por eso la modularidad global
    es moderada y la estructura se aprecia mejor en la red de similitud de Girvan–Newman.
    """)

    titulo("Conclusión del paso 3")
    nota(f"""
    - Girvan–Newman (red de 80 películas): {len(comunidades_gn)} comunidades, Q = {mejor['modularidad']:.2f}; interpretables como culto/Star Wars, taquilla 1995-96, cartelera 1997, populares familiares y clásicos.
    - Louvain (grafo completo): {len(comunidades_lv)} comunidades, Q = {q_lv:.2f}; estructura real pero difusa.
    - Hallazgo transversal: la época organiza la red tanto como el género. El filtrado colaborativo (Parte 2) capta estos patrones latentes sin usar metadatos.

    Siguiente: python pasos\\paso4_modelos_svd_knn.py
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 3: comunidades y visualización")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    args = parser.parse_args()
    with registrar_salida("paso3_comunidades_visualizacion"):
        main(args.mostrar)


if __name__ == "__main__":
    cli()
