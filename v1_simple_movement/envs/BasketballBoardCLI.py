import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
import board

class BasketballBoardCLI(gym.Env):
    """
    Entorno de baloncesto para el CLI.
    Este entorno simula un tablero de baloncesto y permite a los jugadores moverse y lanzar el balón.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self) -> None:
        """
        Inicializa el entorno de baloncesto.
        Define el espacio de acción y el espacio de observación.
        """
        super().__init__()
        self.rows = 15
        self.cols = 28
        self.action_space = spaces.Discrete(5)
        
        # Restricción a mitad ofensiva: columnas 0–13
        low = np.array([0, 0, 0, 0])
        high = np.array([self.rows - 1, 13, self.rows - 1, 13])
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

        self.player_positions = [
            {"pos": [2, 7], "team": 1},
            {"pos": [4, 1], "team": 1},
            {"pos": [10, 3], "team": 1},
            {"pos": [7, 10], "team": 1},
            {"pos": [13, 5], "team": 1},
            {"pos": [2, 7], "team": 2},
            {"pos": [4, 9], "team": 2},
            {"pos": [10, 8], "team": 2},
            {"pos": [7, 12], "team": 2},
            {"pos": [13, 11], "team": 2},
        ]

        self.current_player_idx = 5
        self.ball_pos = self.player_positions[self.current_player_idx]["pos"].copy()
        self.score_blue = 0
        self.score_red = 0
        self.done = False

    def _get_obs(self) -> np.ndarray:
        """
        Devuelve la observación actual del entorno.
        La observación incluye la posición del jugador actual y la posición del balón.
        :return: Array numpy con la posición del jugador y la posición del balón.
        """
        player_pos = self.player_positions[self.current_player_idx]["pos"]
        return np.array(player_pos + self.ball_pos, dtype=np.int32)
    
    def _is_cell_occupied(self, pos) -> bool:
        '''
        Verifica si la celda está ocupada por otro jugador.
        :param pos: Posición a verificar (fila, columna).
        :return: True si la celda está ocupada, False en caso contrario.
        '''
        for i, player in enumerate(self.player_positions):
            if i != self.current_player_idx and player["pos"] == list(pos):
                return True
        return False

    def reset(self, seed=None, options=None) -> tuple:
        """
        Reinicia el entorno y devuelve la observación inicial.
        :param seed: Semilla para la aleatorieda (no utilizada en este caso).
        :param options: Opciones adicionales (no utilizadas).
        :return: Tupla (observación inicial, información adicional).
        """
        super().reset(seed=seed)

        posiciones = [
            [5, 3], [6, 3], [7, 3], [8, 3], [9, 3],
            [5, 13], [6, 13], [7, 13], [8, 13], [9, 13]
        ]
        for i, pos in enumerate(posiciones):
            self.player_positions[i]["pos"] = pos.copy()

        self.current_player_idx = 5
        self.ball_pos = self.player_positions[self.current_player_idx]["pos"].copy()
        self.score_red = 0
        self.score_blue = 0
        self.done = False

        return self._get_obs(), {}

    def step(self, action) -> tuple:
        '''
        Realiza un paso en el entorno con la acción dada.
        :param action: Acción a realizar (0=↑, 1=↓, 2=←, 3=→, 4=lanzar).
        :return: Tupla (observación, recompensa, terminado, truncado, información).
        '''
        reward = 0
        player_pos = self.player_positions[self.current_player_idx]["pos"]
        target_row, target_col = player_pos

        if action == 0 and player_pos[0] > 0:
            target_row -= 1
        elif action == 1 and player_pos[0] < self.rows - 1:
            target_row += 1
        elif action == 2 and player_pos[1] > 0:
            target_col -= 1
        elif action == 3 and player_pos[1] < 13:
            target_col += 1
        elif action == 4:
            r, c = player_pos
            reward = board.get_zone_score(r, c)
            self.score_red += reward
            self.done = True

        if action in [0, 1, 2, 3]:
            if not self._is_cell_occupied((target_row, target_col)):
                player_pos[0], player_pos[1] = target_row, target_col
                self.ball_pos = player_pos.copy()

        obs = self._get_obs()
        terminated = self.done
        truncated = False
        info = {
            "player_position": tuple(player_pos),
            "ball_position": tuple(self.ball_pos),
            "score_red": self.score_red,
            "score_blue": self.score_blue
        }

        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass  # No visualización

    def close(self):
        pass  # Nada que cerrar

if __name__ == "__main__":
    modo = input("\n¿Modo de ejecución? [m]anual / [a]leatorio: ").strip().lower()

    env = BasketballBoardCLI()
    obs, _ = env.reset()
    done = False

    if modo == "a":

        step = 0

        print("\nEpisodio automático (acciones aleatorias)\n")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            print(f"[{step:02d}] Acción: {action} | Recompensa: {reward} | Jugador: {info['player_position']} | Balón: {info['ball_position']}")

        print(f"\nFin del episodio en {step} pasos.")
        print(f"Resultado final → 🔵 {info['score_blue']} | 🔴 {info['score_red']}")
    else:

        print("\nEntorno BasketballBoardEnvCLI iniciado.")
        print("Controles: 0=↑, 1=↓, 2=←, 3=→, 4=Lanzar\n")

        while not done:
            print(f"\nObs: {obs}")
            action = input("Introduce acción (0–4): ")
            if action not in {"0", "1", "2", "3", "4"}:
                print("Acción inválida. Usa 0–4.")
                continue

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            print(f"→ Acción ejecutada: {action}")
            print(f"→ Recompensa: {reward}")
            print(f"→ Pos. jugador: {info['player_position']} | Pos. balón: {info['ball_position']}")
            print(f"→ Puntuación: 🔵 {info['score_blue']} - 🔴 {info['score_red']}")

        print("\nEpisodio finalizado.")

