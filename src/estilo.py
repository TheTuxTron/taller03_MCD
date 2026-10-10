"""Estilo común de gráficos (paleta categórica fija, ejes discretos).

Por defecto los gráficos solo se guardan como PNG en results/figures/.
Con mostrar=True además se abren en una ventana (el script se pausa hasta cerrarla).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .consola import ruta_relativa

# Orden categórico fijo: el color sigue a la entidad, nunca al ranking.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL, NARANJA, AQUA = PALETA[0], PALETA[1], PALETA[2]
GRIS_TEXTO = "#52514e"
GRIS_SUAVE = "#c9c8c2"

_MOSTRAR = False


def aplicar_estilo(mostrar: bool = False) -> None:
    global _MOSTRAR
    _MOSTRAR = mostrar
    if not mostrar:
        plt.switch_backend("Agg")  # sin ventanas: solo se generan los archivos PNG
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRIS_SUAVE,
            "axes.labelcolor": GRIS_TEXTO,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "grid.color": "#ecebe7",
            "grid.linewidth": 0.8,
            "xtick.color": GRIS_TEXTO,
            "ytick.color": GRIS_TEXTO,
            "axes.prop_cycle": mpl.cycler(color=PALETA),
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def guardar(fig, ruta) -> None:
    fig.savefig(ruta)
    print(f"  [gráfico] {ruta_relativa(ruta)}")
    if _MOSTRAR:
        plt.show()
    plt.close(fig)


__all__ = ["PALETA", "AZUL", "NARANJA", "AQUA", "GRIS_TEXTO", "GRIS_SUAVE", "aplicar_estilo", "guardar", "plt"]
