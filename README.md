# Taller III — Análisis de Redes y Sistemas de Recomendación

**Maestría en Ciencia de Datos · Universidad Yachay Tech**
Autor: Joel

Análisis del dataset **MovieLens 100K** desde dos ángulos:

1. **Análisis de redes** (`networkx`): grafo bipartito usuario–película, centralidad, agrupamiento y detección de comunidades.
2. **Sistemas de recomendación** (`surprise`): filtrado colaborativo con SVD y KNN, ajuste de hiperparámetros y Top-5 de recomendaciones.

El proyecto son **scripts de Python** que se ejecutan desde la terminal: imprimen los resultados en la consola y guardan gráficos y tablas en `results\`.

## Resultados principales

| Reto | Resultado |
|---|---|
| 5 usuarios que califican más películas | **405** (737), **655** (685), **13** (636), **450** (540), **276** (518) |
| Película más central | **Star Wars (1977)**: 1.ª en grado y vector propio, 5.ª en intermediación |
| Películas "puente" (intermediación) | Estrenos de 1996–97: *Contact, The English Patient, Scream, Liar Liar* |
| Comunidades | Girvan–Newman: 5 comunidades, modularidad **Q = 0.61** (clásicos, culto/Star Wars, populares familiares, taquilla 1995–96, cartelera 1997). Louvain (grafo completo): 4 comunidades, Q = 0.26 |
| SVD vs. KNN (RMSE en test) | **SVD ajustado 0.912** · KNNBaseline ajustado 0.929 · KNN por defecto 0.934 · SVD por defecto 0.938 |
| Mejores hiperparámetros SVD | `n_factors=50, n_epochs=40, lr_all=0.01, reg_all=0.1` |
| Top-5 para el usuario 196 (SVD) | *Shawshank Redemption, A Close Shave, Schindler's List, Rear Window, Citizen Kane* |

## Estructura del proyecto

```
taller3-redes-recomendacion/
├── main.py                                  # Ejecuta los 6 pasos en orden
├── pasos/
│   ├── paso1_preparacion_datos.py           # Descarga, resumen del dataset, partición train/test
│   ├── paso2_red_centralidad_clustering.py  # Grafo bipartito, grado, centralidad, agrupamiento
│   ├── paso3_comunidades_visualizacion.py   # Girvan–Newman, Louvain y visualizaciones
│   ├── paso4_modelos_svd_knn.py             # SVD vs. KNN con parámetros por defecto
│   ├── paso5_ajuste_hiperparametros.py      # GridSearchCV y evaluación final en test
│   └── paso6_recomendaciones_top5.py        # Top-5 para un usuario específico
├── src/
│   ├── datos.py            # Rutas, descarga y carga de MovieLens
│   ├── red.py              # Construcción del grafo y funciones de comunidades
│   ├── recomendacion.py    # Conversión a surprise y generación de Top-N
│   ├── consola.py          # Formato de la salida en la terminal
│   └── estilo.py           # Estilo común de gráficos
├── results/                # (se genera al ejecutar)
│   ├── figures/            # Gráficos PNG
│   ├── tables/             # Tablas de resultados (CSV)
│   ├── salidas/            # Copia de lo que imprimió cada paso (.txt)
│   └── mejores_parametros.json
├── docs/
│   └── GUION_EXPOSICION.md # Guion para la exposición en clase
├── .vscode/settings.json   # Indica a VS Code que use el intérprete de .venv
├── data/                   # (no se versiona) ml-100k/ y processed/
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo (Windows + Visual Studio Code)


### Paso 1 — Preparar el entorno (una sola vez)

```powershell

# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activarlo (al inicio de la línea debe aparecer "(.venv)")
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si el paso 3 muestra *"la ejecución de scripts está deshabilitada en este sistema"*, ejecutar una vez
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, responder **S** y repetir el paso 3.
>

### Paso 2 — Ejecutar

Con el entorno activado (`(.venv)` al inicio de la línea):

```powershell
python main.py
```


### Dónde quedan los resultados

- `results\figures\` — los 10 gráficos (se abren con doble clic desde el explorador de VS Code).
- `results\tables\` — tablas en CSV.
- `results\salidas\` — copia en texto de todo lo que imprimió cada paso.

### Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `'py' no se reconoce…` / `'python' no se reconoce…` | Reinstalar Python marcando *Add python.exe to PATH*, cerrar y volver a abrir VS Code. |
| `ModuleNotFoundError: No module named 'networkx'` (u otra librería) | El entorno no está activado: ejecutar `.\.venv\Scripts\Activate.ps1` y repetir. |
| Error al instalar `scikit-surprise` | Revisar que Python sea de **64 bits** y versión 3.10–3.13: para esas versiones hay binarios listos y no hace falta compilador. |
| `FileNotFoundError` en el paso 2 o posteriores | Falta ejecutar antes el paso 1 (genera `data\processed\`). |
| El paso 6 dice que falta `mejores_parametros.json` | Falta ejecutar antes el paso 5. |
| La descarga del dataset falla (proxy o firewall) | Descargar a mano `ml-100k.zip` desde <https://grouplens.org/datasets/movielens/100k/> y descomprimir `u.data` y `u.item` en `data\ml-100k\`. |

## Decisiones metodológicas

- **Girvan–Newman no escala** al grafo completo (100 000 aristas; recalcula la intermediación después de cada eliminación). Se aplica sobre una red de similitud de las 80 películas más populares, con peso *lift* de co-valoración (corrige el sesgo de popularidad) y 4 vecinos por película. Louvain se usa como método escalable sobre el grafo completo.
- **Agrupamiento bipartito:** el coeficiente clásico vale 0 en grafos bipartitos (no hay triángulos), por eso se usa el de Latapy et al. (`bipartite.clustering`).
- **Intermediación aproximada** con muestra de 300 nodos fuente (semilla fija) y **vector propio** con `eigenvector_centrality_numpy` (la iteración de potencias no converge en grafos bipartitos).
- **Evaluación sin fuga de datos:** partición fija 80/20; el grid search usa solo el 80 % y el RMSE final se mide en el 20 % reservado.
- **Top-N con umbral de evidencia:** se recomiendan solo películas con ≥ 20 calificaciones, porque sin filtro KNN sugiere películas con una sola valoración (nota estimada 5.0, poco confiable).

## Datos

F. Maxwell Harper y Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5(4). <https://grouplens.org/datasets/movielens/100k/>
