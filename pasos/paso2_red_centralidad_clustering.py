"""PASO 2 · Grafo bipartito: centralidad y agrupamiento  (Parte 1 — networkx)

Grafo bipartito G = (U ∪ M, E): usuarios U, películas M y una arista (u, m)
si el usuario u calificó la película m.

Retos que resuelve:
    - Los 5 usuarios que califican más películas.
    - Las películas más centrales de la red (grado, vector propio, intermediación).
    - Coeficiente de agrupamiento (versión bipartita).

Ejecución:  python pasos\\paso2_red_centralidad_clustering.py [--mostrar]
Duración aproximada: 1-2 minutos (el agrupamiento bipartito es lo más lento).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from networkx.algorithms import bipartite  # noqa: E402

from src.consola import nota, registrar_salida, seccion, tabla, titulo  # noqa: E402
from src.datos import DIR_FIGURAS, DIR_TABLAS, SEMILLA, cargar_procesados  # noqa: E402
from src.estilo import AZUL, NARANJA, aplicar_estilo, guardar, plt  # noqa: E402
from src.red import construir_grafo_bipartito, id_de_nodo, particiones  # noqa: E402


def main(mostrar: bool = False) -> None:
    aplicar_estilo(mostrar)
    titulo("Paso 2 · Grafo bipartito: centralidad y agrupamiento")
    ratings, peliculas = cargar_procesados()
    titulo_de = dict(zip(peliculas["item_id"], peliculas["titulo"]))

    # 1. Construcción del grafo ---------------------------------------------
    seccion("1. Construcción del grafo")
    G = construir_grafo_bipartito(ratings)
    usuarios, pelis = particiones(G)
    print("¿Es bipartito?:", nx.is_bipartite(G))
    print("¿Es conexo?   :", nx.is_connected(G))
    resumen = pd.Series({
        "Nodos usuario": len(usuarios),
        "Nodos película": len(pelis),
        "Aristas (calificaciones)": f"{G.number_of_edges():,}",
        "Densidad bipartita |E| / (|U|·|M|)": round(bipartite.density(G, usuarios), 4),
        "Grado medio usuario": round(sum(d for _, d in G.degree(usuarios)) / len(usuarios), 1),
        "Grado medio película": round(sum(d for _, d in G.degree(pelis)) / len(pelis), 1),
    }, name="valor")
    tabla(resumen.to_frame())

    # 2. Centralidad de grado y reto de usuarios ----------------------------
    seccion("2. Centralidad de grado")
    nota("""
    El grado de un usuario es cuántas películas calificó (actividad); el de una película, cuántos usuarios
    la calificaron (popularidad). La centralidad de grado bipartita lo normaliza por el tamaño del conjunto
    opuesto: C(u) = grado(u) / |M| y C(m) = grado(m) / |U|.
    """)
    cd = bipartite.degree_centrality(G, usuarios)

    tabla_usuarios = pd.DataFrame({
        "user_id": [id_de_nodo(n) for n in usuarios],
        "peliculas_calificadas": [G.degree(n) for n in usuarios],
        "centralidad_grado": [cd[n] for n in usuarios],
    }).sort_values("peliculas_calificadas", ascending=False)

    top5 = tabla_usuarios.head(5).reset_index(drop=True)
    top5.index += 1
    top5.to_csv(DIR_TABLAS / "top5_usuarios_activos.csv", index_label="puesto")
    tabla(top5, ">>> RETO: los 5 usuarios que califican más películas",
          formatters={"centralidad_grado": "{:.1%}".format})

    tabla_pelis = pd.DataFrame({
        "item_id": [id_de_nodo(n) for n in pelis],
        "num_calificaciones": [G.degree(n) for n in pelis],
        "centralidad_grado": [cd[n] for n in pelis],
    })
    tabla_pelis["titulo"] = tabla_pelis["item_id"].map(titulo_de)
    top10 = tabla_pelis.sort_values("num_calificaciones", ascending=False).head(10).reset_index(drop=True)
    top10.index += 1
    tabla(top10[["titulo", "num_calificaciones", "centralidad_grado"]],
          "Las 10 películas más populares (mayor grado):",
          formatters={"centralidad_grado": "{:.1%}".format})

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, nodos, color, nombre in [(ejes[0], usuarios, AZUL, "usuarios"), (ejes[1], pelis, NARANJA, "películas")]:
        ax.hist([G.degree(n) for n in nodos], bins=50, color=color, edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_title(f"Distribución de grado — {nombre}")
        ax.set_xlabel("Grado")
        ax.set_ylabel("Número de nodos (escala log)")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "02_distribucion_grado.png")

    u1 = top5.iloc[0]
    nota(f"""
    Ambas distribuciones son muy asimétricas (cola larga). El usuario más activo (id {int(u1.user_id)}) calificó
    {int(u1.peliculas_calificadas)} películas, el {u1.centralidad_grado:.0%} del catálogo, frente a unas
    {resumen['Grado medio usuario']:.0f} del usuario típico. Unas pocas películas concentran muchas aristas:
    son los "hubs" de la red.
    """)

    # 3. Películas más centrales --------------------------------------------
    seccion("3. ¿Cuáles son las películas más centrales?")
    nota("""
    - Grado: cuántos usuarios la calificaron (popularidad directa).
    - Vector propio: importancia por estar conectada a nodos importantes. Se usa eigenvector_centrality_numpy porque en grafos bipartitos la iteración de potencias oscila y no converge.
    - Intermediación: cuántos caminos más cortos pasan por ella (películas "puente"). Se aproxima con 300 nodos fuente (semilla fija) porque el cálculo exacto es O(|V|·|E|).
    Al final se combinan las tres posiciones en un ranking promedio.
    """)
    t0 = time.time()
    eig = nx.eigenvector_centrality_numpy(G)
    print(f"\nVector propio calculado en {time.time() - t0:.1f} s")
    t0 = time.time()
    btw = nx.betweenness_centrality(G, k=300, seed=SEMILLA)
    print(f"Intermediación (aprox. k=300) calculada en {time.time() - t0:.1f} s")

    tabla_pelis["eigenvector"] = [eig[n] for n in pelis]
    tabla_pelis["betweenness"] = [btw[n] for n in pelis]
    for col in ["centralidad_grado", "eigenvector", "betweenness"]:
        tabla_pelis[f"rank_{col}"] = tabla_pelis[col].rank(ascending=False, method="min").astype(int)
    tabla_pelis["rank_promedio"] = tabla_pelis[["rank_centralidad_grado", "rank_eigenvector", "rank_betweenness"]].mean(axis=1)

    centrales = tabla_pelis.sort_values("rank_promedio").head(10).reset_index(drop=True)
    centrales.index += 1
    centrales.to_csv(DIR_TABLAS / "peliculas_mas_centrales.csv", index_label="puesto")
    vista = centrales[["titulo", "num_calificaciones", "rank_centralidad_grado", "rank_eigenvector", "rank_betweenness"]]
    vista.columns = ["titulo", "calificaciones", "puesto_grado", "puesto_vector_propio", "puesto_intermediacion"]
    tabla(vista, ">>> RETO: las 10 películas más centrales (ranking promedio de las 3 métricas)")

    def top(col, n=5):
        return tabla_pelis.sort_values(col, ascending=False).head(n)["titulo"].tolist()

    tabla(pd.DataFrame({
        "Top por grado": top("centralidad_grado"),
        "Top por vector propio": top("eigenvector"),
        "Top por intermediación": top("betweenness"),
    }, index=range(1, 6)), "Top-5 según cada métrica:")

    corr = tabla_pelis[["centralidad_grado", "eigenvector", "betweenness"]].corr(method="spearman")
    tabla(corr.round(3), "Correlación de Spearman entre las tres métricas (películas):")

    fila = tabla_pelis.set_index("titulo")
    sw, ct = fila.loc["Star Wars (1977)"], fila.loc["Contact (1997)"]
    nota(f"""
    - A nivel global las tres métricas están muy correlacionadas (Spearman entre {corr.values[corr.values < 1].min():.3f} y {corr.values[corr.values < 1].max():.3f}): una película popular suele ser central en cualquier sentido.
    - La diferencia está en la cima. El vector propio premia a los clásicos (Return of the Jedi, Raiders of the Lost Ark, Pulp Fiction) porque los ven los usuarios más activos; la intermediación la lideran los estrenos de 1996-97 (Contact, The English Patient, Scream, Liar Liar).
    - Caso revelador: Contact (1997) es {int(ct.rank_centralidad_grado)}.ª en grado y {int(ct.rank_betweenness)}.ª en intermediación, pero {int(ct.rank_eigenvector)}.ª en vector propio. La vieron muchos usuarios nuevos (llegaron en 1997-98 y calificaron lo que estaba en cartelera), poco conectados con el núcleo: Contact es el PUENTE entre esa cohorte y el resto de la red.
    - Star Wars (1977) es la película más central: {int(sw.rank_centralidad_grado)}.ª en grado, {int(sw.rank_eigenvector)}.ª en vector propio y {int(sw.rank_betweenness)}.ª en intermediación.
    """)

    # 4. Agrupamiento -------------------------------------------------------
    seccion("4. Coeficiente de agrupamiento")
    print("Agrupamiento clásico promedio:", nx.average_clustering(G))
    nota("""
    El coeficiente clásico cuenta triángulos, y en un grafo bipartito no puede haberlos (un ciclo de 3 uniría
    dos nodos del mismo tipo): vale 0 siempre. Por eso se usa el agrupamiento bipartito de Latapy et al. (2008):
    para dos nodos del mismo tipo mide |N(u) ∩ N(v)| / |N(u) ∪ N(v)|, es decir, cuánto se parecen sus
    historiales. Tarda alrededor de 1 minuto...
    """)
    t0 = time.time()
    clus = bipartite.clustering(G, mode="dot")
    print(f"\nAgrupamiento bipartito calculado en {time.time() - t0:.1f} s")

    tabla_usuarios["clustering"] = tabla_usuarios["user_id"].map(lambda u: clus[f"u{u}"])
    tabla_pelis["clustering"] = [clus[n] for n in pelis]
    resumen_clus = pd.DataFrame({
        "promedio": [tabla_usuarios["clustering"].mean(), tabla_pelis["clustering"].mean()],
        "mediana": [tabla_usuarios["clustering"].median(), tabla_pelis["clustering"].median()],
        "máximo": [tabla_usuarios["clustering"].max(), tabla_pelis["clustering"].max()],
    }, index=["Usuarios", "Películas"]).round(4)
    tabla(resumen_clus, "Agrupamiento bipartito:")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].scatter(tabla_usuarios["peliculas_calificadas"], tabla_usuarios["clustering"],
                    s=10, alpha=0.5, color=AZUL, edgecolors="none")
    ejes[0].set_title("Usuarios: agrupamiento vs. grado")
    ejes[0].set_xlabel("Películas calificadas (grado)")
    ejes[0].set_ylabel("Agrupamiento bipartito")
    ejes[1].scatter(tabla_pelis["num_calificaciones"], tabla_pelis["clustering"],
                    s=10, alpha=0.5, color=NARANJA, edgecolors="none")
    ejes[1].set_title("Películas: agrupamiento vs. grado")
    ejes[1].set_xlabel("Calificaciones recibidas (grado)")
    ejes[1].set_ylabel("Agrupamiento bipartito")
    fig.tight_layout()
    guardar(fig, DIR_FIGURAS / "02_clustering_vs_grado.png")

    r_u = tabla_usuarios[["peliculas_calificadas", "clustering"]].corr("spearman").iloc[0, 1]
    r_m = tabla_pelis[["num_calificaciones", "clustering"]].corr("spearman").iloc[0, 1]
    print(f"\nCorrelación de Spearman grado vs. agrupamiento: usuarios {r_u:.3f} | películas {r_m:.3f}")
    nota(f"""
    - Los valores son bajos (≈{resumen_clus.loc['Usuarios', 'promedio']:.2f} usuarios, ≈{resumen_clus.loc['Películas', 'promedio']:.2f} películas): dos usuarios que comparten una película comparten en promedio menos del 10 % de su historial. La red es dispersa y heterogénea: los gustos son personales, lo que justifica el filtrado colaborativo frente a "recomendar lo más popular a todos".
    - La relación con el grado es positiva. Los usuarios muy activos han visto buena parte de lo que ven los demás: forman el NÚCLEO de la red. Los usuarios con pocas películas tienen historiales particulares que casi no se solapan: están en la PERIFERIA. Parte del efecto es mecánico (con más películas es más fácil coincidir), por eso el agrupamiento se interpreta junto al grado.
    """)

    titulo("Conclusión del paso 2")
    usuarios_txt = ", ".join(f"{r.user_id} ({r.peliculas_calificadas})" for r in top5.itertuples())
    nota(f"""
    - 5 usuarios más activos: {usuarios_txt}.
    - Película más central: Star Wars (1977).
    - Películas puente: estrenos de 1996-97 (Contact, The English Patient, Scream).
    - Agrupamiento bajo y creciente con el grado: núcleo de usuarios activos, periferia de usuarios ocasionales.

    Siguiente: python pasos\\paso3_comunidades_visualizacion.py
    """)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paso 2: centralidad y agrupamiento del grafo bipartito")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    args = parser.parse_args()
    with registrar_salida("paso2_red_centralidad_clustering"):
        main(args.mostrar)


if __name__ == "__main__":
    cli()
