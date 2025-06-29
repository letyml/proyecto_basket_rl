from stable_baselines3.common.callbacks import BaseCallback

class EpisodeStatsCallback(BaseCallback):
    """
    Callback para guardar estadísticas de episodios completos durante el entrenamiento.
    Guarda información sobre recompensas, longitudes de episodios y datos de cada paso.
    """

    def __init__(self, save_path=None, verbose=0) -> None:
        """
        Inicializa el callback.
        :param save_path: Ruta donde se guardarán los datos de los episodios.
        :param verbose: Nivel de verbosidad (0 = sin salida, 1 = salida básica).
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_data = []
        self.current_episode = []
        self.current_reward = 0
        self.current_length = 0
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        Método que se llama en cada paso del entrenamiento.
        Guarda información sobre el paso actual y actualiza las estadísticas del episodio.
        :return: True para continuar el entrenamiento, False para detenerlo.
        """

        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        action = self.locals["actions"][0]
        info = self.locals["infos"][0] if "infos" in self.locals else {}

        # Extraer probabilidades y estadísticas de defensa
        base_prob, final_prob, defense_penalty, defense_count, defenders = info.get("prob_info", (None, None, None, None, []))

        self.current_episode.append({
            "step": self.num_timesteps,
            "action": action.tolist() if hasattr(action, "tolist") else action if isinstance(action, list) else [action],
            "reward": float(reward),
            "player_idx": info.get("player_idx"),
            "player_name": info.get("player_name"),
            "player_pos": info.get("player_position"),
            "ball_pos": info.get("ball_position"),
            "shot": info.get("shot"),
            "prob_info": {
                "base": base_prob,
                "final": final_prob,
                "penalty": defense_penalty,
                "defense_count": defense_count,
                "defenders": defenders
            },
            "done": bool(done),
            "score_red": info.get("score_red"),
            "score_blue": info.get("score_blue"),
            "all_players": info.get("all_players")
        })

        self.current_reward += reward
        self.current_length += 1

        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_data.append(self.current_episode)
            self.current_episode = []
            self.current_reward = 0
            self.current_length = 0

        return True
