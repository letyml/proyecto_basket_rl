import os
import sys
import time
import pandas as pd

# Añadir rutas
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs")) # entorno
from BasketballBoardVisual import BasketballBoardEnv

def play_episode(csv_path, episodio=0, fps=5, enable_countdown=True) -> None:
    """
    Reproduce visualmente un episodio específico desde un CSV.
    :param csv_path: Ruta al archivo CSV con la información de episodios.
    :param episodio: Número de episodio a reproducir.
    :param fps: Cuadros por segundo para la animación.
    :param enable_countdown: Si True, muestra la cuenta atrás al inicio.
    """
    df = pd.read_csv(csv_path)
    if episodio not in df["episode"].unique():
        print(f" El episodio {episodio} no está presente en el archivo.")
        return

    df_ep = df[df["episode"] == episodio]
    env = BasketballBoardEnv(render_mode="human")
    obs, _ = env.reset(skip_countdown=not enable_countdown) 

    env.score_blue = 0
    env.score_red = 0

    print(f"\n🎞 Reproduciendo episodio {episodio}...")

    for _, row in df_ep.iterrows():
        jugador = (int(row["player_row"]), int(row["player_col"]))
        balon = (int(row["ball_row"]), int(row["ball_col"]))
        reward = row["reward"]
        accion = int(row["action"])

        # Actualizar posiciones y marcador
        env.player_positions[env.current_player_idx]["pos"] = list(jugador)
        env.ball_pos = list(balon)
        env.done = row["done"]

        if reward > 0:
            env.score_red += int(reward) 

        print(f"Paso: Acción={accion} | Recompensa={reward} | PosJugador={jugador} | Balón={balon}")
        env.render()
        time.sleep(1 / fps)

    time.sleep(2)
    print("\nFin del episodio.")


if __name__ == "__main__":
    nombre_modelo = input("Nombre del modelo (ej. model_v1_YYYYMMDD_HHMMSS): ").strip()
    ruta_csv = os.path.join(os.path.dirname(__file__), "..", "results", nombre_modelo, "episode_data.csv")

    if not os.path.exists(ruta_csv):
        print("No se encontró el archivo CSV.")
        sys.exit()

    # Preguntar si quiere reproducir varios episodios
    multi = input("¿Quieres reproducir más de un episodio? [s/n]: ").strip().lower()

    if multi == "s":
        episodios_str = input("Introduce los números de episodios separados por comas (ej. 0,1,2): ")
        try:
            episodios = [int(e.strip()) for e in episodios_str.split(",")]
        except ValueError:
            print("Formato inválido. Debes introducir números separados por comas.")
            sys.exit()

        for idx, ep in enumerate(episodios):
            play_episode(ruta_csv, episodio=ep, enable_countdown=(idx == 0))
            time.sleep(1)
    elif multi == "n":
        episodio = int(input(" Número de episodio a visualizar (ej. 0): ").strip())
        play_episode(ruta_csv, episodio=episodio)
    else:
        print("Opción no válida. Debes introducir 's' o 'n'.")
        sys.exit()
