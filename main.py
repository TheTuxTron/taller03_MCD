"""Ejecuta los 6 pasos del Taller III en orden.

Uso (desde la carpeta del proyecto, con el entorno .venv activado):
    python main.py                 ejecuta los pasos 1 a 6
    python main.py --pasos 1 2 3   ejecuta solo los pasos indicados
    python main.py --mostrar       además abre cada gráfico en una ventana
    python main.py --usuario 13    usuario para el Top-5 del paso 6 (por defecto 196)

Cada paso también se puede ejecutar por separado, por ejemplo:
    python pasos\\paso2_red_centralidad_clustering.py
"""
import argparse
import importlib.util
import time
from pathlib import Path

from src.consola import registrar_salida

RAIZ = Path(__file__).resolve().parent
PASOS = {
    1: "paso1_preparacion_datos",
    2: "paso2_red_centralidad_clustering",
    3: "paso3_comunidades_visualizacion",
    4: "paso4_modelos_svd_knn",
    5: "paso5_ajuste_hiperparametros",
    6: "paso6_recomendaciones_top5",
}


def cargar_paso(nombre: str):
    ruta = RAIZ / "pasos" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> None:
    parser = argparse.ArgumentParser(description="Taller III: análisis de redes y sistemas de recomendación")
    parser.add_argument("--pasos", type=int, nargs="+", choices=sorted(PASOS), default=sorted(PASOS),
                        help="pasos a ejecutar (por defecto todos)")
    parser.add_argument("--mostrar", action="store_true", help="abre los gráficos en una ventana además de guardarlos")
    parser.add_argument("--usuario", type=int, default=196, help="usuario para el Top-5 del paso 6")
    args = parser.parse_args()

    inicio = time.time()
    for numero in sorted(args.pasos):
        nombre = PASOS[numero]
        modulo = cargar_paso(nombre)
        t0 = time.time()
        if numero == 6:
            with registrar_salida(f"paso6_recomendaciones_usuario_{args.usuario}"):
                modulo.main(args.usuario)
        else:
            with registrar_salida(nombre):
                modulo.main(args.mostrar)
        print(f"[Paso {numero} terminado en {time.time() - t0:.0f} s]")
    print(f"\nTodo listo en {(time.time() - inicio) / 60:.1f} minutos. Resultados en la carpeta results\\")


if __name__ == "__main__":
    main()
