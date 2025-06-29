# Este script descarga todos los tiros (FGA) de los jugadores seleccionados en la temporada 2024-25 y los guarda en data/ sin ningún filtrado
# Consulta de tiros (shotchartdetail)
# Consulta de la posición oficial (commonplayerinfo)

import os
import time
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail, commonplayerinfo
from nba_api.stats.static import players

# Lista de jugadores a consultar
NBA_PLAYERS = [
    "James Harden",
    "Kris Dunn",
    "Norman Powell",
    "Kawhi Leonard",
    "Ivica Zubac"
]

SEASON_TYPE = "Regular Season"
SEASON = "2024-25"
TEAM_ID_CLIPPERS = 1610612746

# Carpeta donde se guardarán los datos (relativa a este script)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_player_id(full_name) -> int:
    """
    Busca el ID del jugador por su nombre completo.
    :param full_name: Nombre completo del jugador.
    :return: ID del jugador si se encuentra, o None si no existe.
    """
    player = players.find_players_by_full_name(full_name)
    if player:
        return player[0]['id']
    else:
        raise ValueError(f"Jugador no encontrado: {full_name}")

def get_player_position(player_id) -> str:
    """
    Consulta la posición oficial del jugador usando commonplayerinfo.
    :param player_id: ID del jugador.
    :return: Posición del jugador como string.
    """
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    position = info.get_data_frames()[0].loc[0, "POSITION"]
    return position

def get_shot_chart(player_name) -> pd.DataFrame:
    """
    Descarga los datos de tiros del jugador con los Clippers en la temporada especificada.
    :param player_name: Nombre completo del jugador.
    :return: DataFrame con los datos de tiros del jugador.
    """
    player_id = get_player_id(player_name)
    position = get_player_position(player_id)
    print(f"Descargando datos de {player_name} (ID: {player_id})...")

    response = shotchartdetail.ShotChartDetail(
        team_id=TEAM_ID_CLIPPERS,
        player_id=player_id,
        season_type_all_star=SEASON_TYPE,
        season_nullable=SEASON,
        context_measure_simple="FGA" # tiros intentados (FGA, Field Goal Attempts) para luego calcular el porcentaje de acierto
    )
    data = response.get_data_frames()[0]
    data["PLAYER_POSITION"] = position
    return data

def save_all_players_shots(output_dir=OUTPUT_DIR) -> None:
    """
    Guarda los datos de tiros de todos los jugadores en archivos CSV.
    :param output_dir: Directorio donde se guardarán los archivos CSV.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    for name in NBA_PLAYERS:
        try:
            df = get_shot_chart(name)
            safe_name = name.replace(" ", "_").lower()
            file_path = os.path.join(output_dir, f"{safe_name}_shots.csv")
            df.to_csv(file_path, index=False)
            print(f"Guardado: {file_path} ({len(df)} tiros)")
            time.sleep(1.5)
        except Exception as e:
            print(f"Error con {name}: {e}")

if __name__ == "__main__":
    save_all_players_shots()
