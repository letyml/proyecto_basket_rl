import os
import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from datetime import datetime

# ================================
# Importación de módulos propios
# ================================
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))  # callback personalizado
from episodeStatsCallback import EpisodeStatsCallback

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))          # entorno sin renderizado
from BasketballBoardCLI import BasketballBoardCLI

# ================================
# Configuración de directorios
# ================================
results_path = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_path, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
mlruns_path = os.path.join(project_root, "mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("Basketball RL Model V1")  # nombre del experimento

# ==================================
# Hiperparámetros de entrenamiento
# ==================================
total_iters   = 100000         # número total de pasos
learning_rate = 0.0003      # tasa de aprendizaje
buffer_size   = 100000      # tamaño del buffer de experiencia
batch_size    = 32          # tamaño del batch
gamma         = 0.99        # factor de descuento

# ===================================================
# Función auxiliar para guardar episodios completos
# ===================================================
def save_episodes(callback, path_salida_csv) -> None:
    """
    Guarda los episodios completos en un archivo CSV.
    :param callback: Callback que contiene los datos de los episodios.
    :param path_salida_csv: Ruta del archivo CSV de salida.
    """
    if path_salida_csv:
        data = []
        for ep_idx, episodio in enumerate(callback.episode_data):
            for step in episodio:
                data.append({
                    "episode": ep_idx,
                    "step": step["step"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "player_row": step["player_pos"][0] if step["player_pos"] else None,
                    "player_col": step["player_pos"][1] if step["player_pos"] else None,
                    "ball_row": step["ball_pos"][0] if step["ball_pos"] else None,
                    "ball_col": step["ball_pos"][1] if step["ball_pos"] else None,
                    "done": step["done"]
                })
        df_all = pd.DataFrame(data)
        df_all.to_csv(path_salida_csv, index=False)
        print(f"Guardado como CSV en: {path_salida_csv}")

# ===================================
# Inicio de entrenamiento con MLflow
# ===================================
with mlflow.start_run() as run:
    print(f"Ejecutando run_id: {run.info.run_id}")

    # Preparar carpetas de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_path, f"model_v1_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "train_v1")

    mlflow.set_tag("run_name", f"modelo_v1_{timestamp}")

    # Guardar el run_id en un archivo de texto
    run_id_path = os.path.join(run_dir, "mlflow_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(run.info.run_id)

    # Registrar archivo de versión (si existe)
    version_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v1_simple_movement.txt"))
    if os.path.exists(version_info_path):
        mlflow.log_artifact(version_info_path, artifact_path="metadata")
    else:
        print("No se encontró v1_simple_movement.txt en la ruta esperada:", version_info_path)

    # Inicializar entorno y modelo DQN (Deep Q-Network)
    env = BasketballBoardCLI()
    model = DQN(
        "MlpPolicy", # política de red neuronal multi-capa util para entornos discretos (5 acciones fijas)
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1
    )

    stats_callback = EpisodeStatsCallback(save_path=run_dir)

    print("Entrenamiento iniciado...\n")
    start_time = datetime.now()
    model.learn(total_timesteps=total_iters, callback=stats_callback)
    end_time = datetime.now()

    # Guardar modelo entrenado
    model.save(model_path)

    # Extraer métricas de entrenamiento
    rewards = stats_callback.episode_rewards
    lengths = stats_callback.episode_lengths

    if rewards:
        rewards = np.array(rewards)
        lengths = np.array(lengths)
        success_rate = np.sum(rewards > 0) / len(rewards)
        steps_per_success = (
            np.mean([lengths[i] for i in range(len(rewards)) if rewards[i] > 0])
            if np.any(rewards > 0) else 0
        )

        # Log de métricas en MLflow
        mlflow.log_metric("train_reward_mean", np.mean(rewards))
        mlflow.log_metric("train_reward_std", np.std(rewards))
        mlflow.log_metric("train_reward_min", np.min(rewards))
        mlflow.log_metric("train_reward_max", np.max(rewards))
        mlflow.log_metric("train_success_rate", success_rate)
        mlflow.log_metric("train_steps_mean", np.mean(lengths))
        mlflow.log_metric("train_steps_std", np.std(lengths))
        mlflow.log_metric("train_steps_per_success_mean", steps_per_success)

        # Guardar resumen
        df = pd.DataFrame({
            "episode": np.arange(len(rewards)),
            "reward": rewards,
            "length": lengths
        })

        # Suavizar recompensas para visualización
        window_size = max(10, min(50, len(df) // 5))  # Ventana dinámica entre 10 y 50 episodios
        df["reward_smooth"] = df["reward"].rolling(window=window_size).mean().fillna(df["reward"])

        # Gráfico de recompensas
        plt.figure(figsize=(10, 6))
        plt.plot(df["episode"], df["reward_smooth"], label="Recompensa suavizada", color="tab:blue", linewidth=2)
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Evolución de la recompensa por episodio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        graph_path = os.path.join(run_dir, "reward_plot.png")
        plt.savefig(graph_path)
        plt.close()
        mlflow.log_artifact(graph_path, artifact_path="metrics")

    # Guardar episodios completos
    csv_path = os.path.join(run_dir, "episode_data.csv")
    save_episodes(stats_callback, csv_path)
    mlflow.log_artifact(csv_path, artifact_path="metrics")

    # Log de parámetros e información final
    mlflow.log_param("total_iters", total_iters)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("buffer_size", buffer_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("gamma", gamma)
    mlflow.log_artifact(model_path + ".zip", artifact_path="model")

    print(f"\nEntrenamiento completado en {end_time - start_time} y guardado en {model_path}.zip\n")
