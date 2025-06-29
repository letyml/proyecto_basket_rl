# train_agent_v2.py
import os
import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Usamos PPO porque es un algoritmo basado en política que admite espacios de acción MultiDiscrete, 
# permitiendo que un solo agente controle múltiples jugadores simultáneamente, 
# a diferencia de DQN que solo acepta acciones discretas individuales
from stable_baselines3 import PPO
from datetime import datetime

# ================================
# Módulos propios
# ================================
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from episodeStatsCallback import EpisodeStatsCallback

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from BasketballBoardCLI import BasketballBoardCLIV2

# ================================
# Configuración
# ================================
results_path = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_path, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
mlruns_path = os.path.join(project_root, "mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("Basketball RL Model V2")

# Hiperparámetros
total_iters = 100000
learning_rate = 0.0003
batch_size = 32
gamma = 0.99

# Guardar episodios
def save_episodes(callback, path_salida_csv):
    if path_salida_csv:
        data = []
        for ep_idx, episodio in enumerate(callback.episode_data):
            for step in episodio:
                row = {
                    "episode": ep_idx,
                    "step": step["step"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "player_idx": step["player_idx"],
                    "player_row": step["player_pos"][0],
                    "player_col": step["player_pos"][1],
                    "ball_row": step["ball_pos"][0],
                    "ball_col": step["ball_pos"][1],
                    "probability": step.get("probability"),
                    "shot": step["shot"],
                    "done": step["done"],
                    "score_red": step.get("score_red"),
                    "score_blue": step.get("score_blue"),
                }
                for i, p in enumerate(step["all_players"]):
                    row[f"p{i}_row"] = p[0]
                    row[f"p{i}_col"] = p[1]
                data.append(row)
        pd.DataFrame(data).to_csv(path_salida_csv, index=False)
        print(f"Guardado como CSV en: {path_salida_csv}")

# Entrenamiento
with mlflow.start_run() as run:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_path, f"model_v2_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "train_v2")

    mlflow.set_tag("run_name", f"modelo_v2_{timestamp}")

    # Guardar el run_id en un archivo de texto
    run_id_path = os.path.join(run_dir, "mlflow_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(run.info.run_id)

    # Registrar archivo de versión (si existe)
    version_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v2_pass_shot_probability.txt"))
    if os.path.exists(version_info_path):
        mlflow.log_artifact(version_info_path, artifact_path="metadata")
    else:
        print("No se encontró v2_pass_shot_probability.txt en la ruta esperada:", version_info_path)


    env = BasketballBoardCLIV2()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1
    )

    stats_callback = EpisodeStatsCallback(save_path=run_dir)
    print("Entrenamiento iniciado...\n")
    start_time = datetime.now()
    model.learn(total_timesteps=total_iters, callback=stats_callback)
    end_time = datetime.now()

    model.save(model_path)

    rewards = np.array(stats_callback.episode_rewards)
    lengths = np.array(stats_callback.episode_lengths)

    if len(rewards):
        mlflow.log_metric("reward_mean", rewards.mean())
        mlflow.log_metric("reward_std", rewards.std())
        mlflow.log_metric("reward_max", rewards.max())
        mlflow.log_metric("reward_min", rewards.min())

        df = pd.DataFrame({
            "episode": np.arange(len(rewards)),
            "reward": rewards,
            "length": lengths
        })
        df["reward_smooth"] = df["reward"].rolling(window=10).mean().fillna(df["reward"])

        plt.figure(figsize=(10, 6))
        plt.plot(df["episode"], df["reward_smooth"], label="Recompensa suavizada")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Evolución de la recompensa")
        plt.grid(True)
        plt.tight_layout()
        graph_path = os.path.join(run_dir, "reward_plot.png")
        plt.savefig(graph_path)
        mlflow.log_artifact(graph_path)

    csv_path = os.path.join(run_dir, "episode_data.csv")
    save_episodes(stats_callback, csv_path)
    mlflow.log_artifact(csv_path)

    mlflow.log_param("total_iters", total_iters)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("batch_size", batch_size)

    print(f"\nEntrenamiento finalizado. Modelo guardado en: {model_path}.zip")
