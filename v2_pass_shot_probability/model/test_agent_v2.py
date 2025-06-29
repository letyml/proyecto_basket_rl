import os
import sys
import time
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))  # entorno
from BasketballBoardVisual import BasketballBoardVisualV2

def play_episode(csv_path, episodio=0, fps=5, enable_countdown=False, visual=True):
    """
    Reproduce visualmente un episodio específico desde un CSV.
    :param csv_path: Ruta al archivo CSV con la información de episodios.
    :param episodio: Número de episodio a reproducir.
    :param fps: Cuadros por segundo para la animación.
    :param enable_countdown: Si True, muestra la cuenta atrás al inicio.
    :param visual: Si True, activa la visualización.
    """    
    df = pd.read_csv(csv_path)
    if episodio not in df["episode"].unique():
        print(f"El episodio {episodio} no está presente en el archivo.")
        return

    df_ep = df[df["episode"] == episodio]
    env = BasketballBoardVisualV2(render_mode="human")
    obs, _ = env.reset(skip_countdown=not enable_countdown, visual=visual)

    env.score_blue = 0
    env.score_red = 0

    print(f"\n🎞 Reproduciendo episodio {episodio}...")

    for _, row in df_ep.iterrows():
        env.current_player_idx = int(row["player_idx"])
        env.ball_pos = [int(row["ball_row"]), int(row["ball_col"])]
        env.score_red = int(row["score_red"])
        env.score_blue = int(row["score_blue"])
        env.last_shot_status = row["shot"]
        env.done = bool(row["done"])

        for i in range(10):
            env.player_positions[i]["pos"] = [int(row[f"p{i}_row"]), int(row[f"p{i}_col"])]

        print(f"Acción={row['action']} | Jugador={env.current_player_idx} | Recompensa={row['reward']} | Balón={env.ball_pos} | Tiro: {row['shot']} | "
              f"Estado: {row['done']} | Puntaje: Azul={env.score_blue} | Rojo={env.score_red}")
        env.render()
        time.sleep(1 / fps)

    time.sleep(1.5)
    print("\nFin del episodio.")

if __name__ == "__main__":
    nombre_modelo = input("Nombre del modelo (ej. model_v2_YYYYMMDD_HHMMSS): ").strip()
    ruta_csv = os.path.join(os.path.dirname(__file__), "..", "results", nombre_modelo, "episode_data.csv")

    if not os.path.exists(ruta_csv):
        print(" No se encontró el archivo CSV.")
        sys.exit()

    multi = input("¿Quieres reproducir más de un episodio? [s/n]: ").strip().lower()

    if multi == "s":
        episodios_str = input("Introduce los números de episodios separados por comas (ej. 0,1,2): ")
        try:
            episodios = [int(e.strip()) for e in episodios_str.split(",")]
        except ValueError:
            print(" Formato inválido. Debes introducir números separados por comas.")
            sys.exit()

        for idx, ep in enumerate(episodios):
            play_episode(ruta_csv, episodio=ep, enable_countdown=(idx == 0), visual=(idx == 0))
            time.sleep(1)
    elif multi == "n":
        episodio = int(input("Número de episodio a visualizar (ej. 0): ").strip())
        play_episode(ruta_csv, episodio=episodio)
    else:
        print("Opción no válida. Debes introducir 's' o 'n'.")
        sys.exit()