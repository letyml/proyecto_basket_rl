import os
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
ORIGINAL_JSON = os.path.join(DATA_FOLDER, "player_shooting_profiles.json")
FILLED_JSON = os.path.join(DATA_FOLDER, "player_shooting_profiles_knn.json")

NBA_X_MIN, NBA_X_MAX = -250, 250
NBA_Y_MIN, NBA_Y_MAX = -40, 470
ROWS, COLS = 15, 14  # media cancha

# ===========================
# FUNCIONES DE MAPEADO
# ===========================
def map_coords_to_cell(loc_x, loc_y) -> tuple:
    """
    Devuelve la celda (columna, fila) correspondiente a unas coordenadas dadas.
    :param loc_x: Coordenada X en el espacio continuo.
    :param loc_y: Coordenada Y en el espacio continuo.
    :param cell_size: Tamaño de cada celda de la cuadrícula.
    :return: Tupla (columna, fila) con índices enteros de la celda.
    """

    x_norm = (loc_x - NBA_X_MIN) / (NBA_X_MAX - NBA_X_MIN)
    y_norm = (loc_y - NBA_Y_MIN) / (NBA_Y_MAX - NBA_Y_MIN)
    row = int(x_norm * (ROWS - 1))
    col = int(y_norm * (COLS - 1))
    if col > 13:
        return None
    return row, col

def calculate_distance_to_basket(r, c, aro=(7, 0)) -> float:
    """
    Calcula la distancia euclídea desde una posición (r, c) hasta el aro.
    :param r: Fila o coordenada vertical del jugador.
    :param c: Columna o coordenada horizontal del jugador.
    :param aro: Tupla con la posición del aro (fila, columna). Por defecto (7, 0).
    :return: Distancia euclídea (float) hasta el aro.
    """
    return np.sqrt((r - aro[0])**2 + (c - aro[1])**2)

# ===========================
# GENERAR JSON ORIGINAL
# ===========================
shooting_profiles = {}

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith("_shots.csv"):
        player_name = filename.replace("_shots.csv", "")
        df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
        df = df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_ZONE_AREA"])

        cells = []
        for _, row in df.iterrows():
            cell = map_coords_to_cell(float(row["LOC_X"]), float(row["LOC_Y"]))
            if cell and cell != (7, 0):
                cells.append(str(cell))
            else:
                cells.append(None)

        df["cell"] = cells
        df = df.dropna(subset=["cell"])

        stats = {}
        for cell, group in df.groupby("cell"):
            total = len(group)
            made = group["SHOT_MADE_FLAG"].sum()
            prob = round(made / total, 2)
            zone = group["SHOT_ZONE_AREA"].mode()[0]
            stats[cell] = {"prob": max(prob, 0.1), "zone": zone}

        shooting_profiles[player_name] = stats
        print(f"{player_name}: {len(stats)} celdas mapeadas")

with open(ORIGINAL_JSON, "w") as f:
    json.dump(shooting_profiles, f, indent=4)

# ===========================
# APLICAR KNN (con distancia)
# ===========================
filled_profiles = {}

for player, profile in shooting_profiles.items():
    known_X = []
    known_y = []

    for key, val in profile.items():
        r, c = eval(key)
        dist = calculate_distance_to_basket(r, c)
        known_X.append([r, c, dist])
        known_y.append(val["prob"])

    knn = KNeighborsRegressor(n_neighbors=min(10, len(known_X)))
    knn.fit(known_X, known_y)

    full_profile = {}
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == (7, 0):  # No estimar en la celda del aro
                continue

            key = str((r, c))
            dist = calculate_distance_to_basket(r, c)

            if key in profile:
                full_profile[key] = profile[key]
            else:
                prob = round(knn.predict([[r, c, dist]])[0], 2)
                full_profile[key] = {"prob": prob, "zone": "Unknown"}

    filled_profiles[player] = full_profile

with open(FILLED_JSON, "w") as f:
    json.dump(filled_profiles, f, indent=4)
