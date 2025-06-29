import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import pygame 
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
import board

class BasketballBoardBaseEnvV4(gym.Env):
    """
    Entorno de baloncesto para visualización en Pygame y CLI.
    Este entorno simula un tablero de baloncesto en el que los jugadores pueden moverse, lanzar a canasta y, 
    en el caso del jugador que tiene la posesión del balón, realizar pases a sus compañeros.
    El equipo rival puede interceptar tiros y pases.
    Los jugadores cuentan con roles y probabilidad de acierto calculada con datos de la nba.
    """
    def __init__(self) -> None:
        """
        Inicializa el entorno de baloncesto.
        Define el espacio de acción y el espacio de observación.
        """
        super().__init__()
        self.rows = 15
        self.cols = 28
        self.action_space = spaces.MultiDiscrete([6] * 5)
        
        # Restricción a mitad ofensiva: columnas 0–13
        self.observation_space = spaces.Box(
            low=np.array([[0, 0]] * 10),
            high=np.array([[self.rows - 1, 13]] * 10),
            dtype=np.int32
        )

        with open(os.path.join(os.path.dirname(__file__), "..", "nba_api", "data", "player_shooting_profiles_knn.json"), "r") as f:
            self.shooting_profiles = json.load(f)

    def _get_obs(self) -> np.ndarray:
        """
        Devuelve la observación actual del entorno.
        La observación incluye la posición del jugador actual y la posición del balón.
        :return: Array numpy con la posición del jugador y la posición del balón.
        """
        return np.array([p["pos"] for p in self.player_positions], dtype=np.int32)

    def _get_valid_actions(self, player_idx) -> list:
        """
        Devuelve una lista de acciones válidas para el jugador.
        :param player_idx: Índice del jugador.
        :return: Lista de acciones válidas.
        """
        if self.player_positions[player_idx].get("has_ball", False):
            return [0, 1, 2, 3, 4, 5] # Si tiene el balón, puede moverse, pasar o lanzar
        return [0, 1, 2, 3] # Si no tiene el balón, solo puede moverse

    def _is_cell_occupied(self, pos) -> bool:
        """
        Verifica si una celda está ocupada por otro jugador.
        :param pos: Posición a verificar (fila, columna).
        :return: True si la celda está ocupada, False en caso contrario.
        """
        return any(p["pos"] == list(pos) for p in self.player_positions)

    def _countdown(self) -> None:
        """
        Muestra una cuenta atrás de 5 segundos antes de iniciar el juego en la visualización de Pygame.
        """
        pygame.init()
        font = pygame.font.SysFont("Arial", 100, bold=True)
        anim_players = [
            {"pos": [2, 7], "color": board.BLUE},
            {"pos": [4, 1], "color": board.BLUE},
            {"pos": [10, 3], "color": board.BLUE},
            {"pos": [7, 10], "color": board.BLUE},
            {"pos": [13, 5], "color": board.BLUE},
            {"pos": [2, 20], "color": board.RED},
            {"pos": [4, 26], "color": board.RED},
            {"pos": [10, 24], "color": board.RED},
            {"pos": [7, 17], "color": board.RED},
            {"pos": [13, 22], "color": board.RED},
        ]
        for i in range(5, 0, -1):
            board.draw_court(self.screen)
            board.draw_players(self.screen, anim_players, current_player_idx=5)
            board.draw_ball_absolute(self.screen, [board.WIDTH // 2, (board.HEIGHT - 50) // 2])
            board.draw_score(self.screen, 0, 0)
            text = font.render(str(i), True, board.WHITE)
            self.screen.blit(text, (board.WIDTH // 2 - 30, board.HEIGHT // 2 - 80))
            pygame.display.flip()
            pygame.time.wait(1000)

    def _get_shot_probability(self, row, col) -> tuple:
        """
        Calcula la probabilidad de éxito de un tiro a canasta en función de la posición del jugador y la defensa.
        :param row: Fila del jugador que lanza.
        :param col: Columna del jugador que lanza.
        :return: Tupla con la probabilidad base, la probabilidad final, la penalización de defensa, el número de defensores cercanos y sus íds
        """
        shooter = self.player_positions[self.current_player_idx]
        shooter_name = shooter.get("name", "").lower().replace(" ", "_")
        shooter_color = shooter["color"]
        opponent_color = board.BLUE if shooter_color == board.RED else board.RED

        cell_key = str((row, col))
        base_prob = self.shooting_profiles.get(shooter_name, {}).get(cell_key, {}).get("prob", 0.1)

        defense_penalty = 0
        defense_count = 0
        defenders = []

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    for idx, player in enumerate(self.player_positions):
                        if player["pos"] == [r, c] and player["color"] == opponent_color:
                            manhattan = abs(dr) + abs(dc)
                            penalty = 0.2 if manhattan == 1 else 0.1 if manhattan == 2 else 0
                            if penalty > 0:
                                defense_penalty += penalty
                                defense_count += 1
                                defenders.append((idx, player["pos"].copy()))
                            break

        final_prob = max(base_prob - defense_penalty, 0.05)
        return base_prob, final_prob, defense_penalty, defense_count, defenders

    def _calculate_shot_reward(self, row, col) -> float:
        """
        Determina si el tiro es exitoso y calcula la recompensa asociada.
        Utiliza la probabilidad base de la celda, que puede ser ajustada en versiones futuras.
        :param row: Fila del jugador que lanza.
        :param col: Columna del jugador que lanza.
        :return: Recompensa recibida tras el tiro.
        """
        _, prob, _, _, _ = self._get_shot_probability(row, col)

        if random.random() < prob:
            base_score = board.get_zone_score(row, col)
            reward = base_score * prob
            self.score_red += base_score
            self.last_shot_status = 'shot_successful'
            self.done = True
        else:
            reward = 0
            self.last_shot_status = 'shot_failed'
            self.done = True

        return reward
    
    def _is_intercepted(self, pos, intercept_prob=0.4, max_manhattan=1) -> bool:
        for j in range(5):  # jugadores azules
            def_r, def_c = self.player_positions[j]["pos"]
            dist = abs(def_r - pos[0]) + abs(def_c - pos[1])
            if 1 <= dist <= max_manhattan:
                prob = intercept_prob / dist
                if random.random() < prob:
                    return True
        return False

    def _build_info(self) -> dict:
        """
        Construye un diccionario con información adicional sobre el estado del entorno.
        :return: Diccionario con información adicional.
        """
        player = self.player_positions[self.current_player_idx]
        player_pos = player["pos"]
        return {
            "player_position": tuple(player_pos),
            "player_idx": self.current_player_idx,
            "player_name": player.get("name", ""),
            "ball_position": tuple(self.ball_pos),
            "score_red": self.score_red,
            "score_blue": self.score_blue,
            "prob_info": self._get_shot_probability(*player_pos),
            "shot": self.last_shot_status,
            "all_players": [p["pos"].copy() for p in self.player_positions]
        }

    
    def reset(self, seed=None, options=None, skip_countdown=False, visual = True) -> tuple:
        """
        Reinicia el entorno de baloncesto.
        :param seed: Semilla para la aleatoriedad.
        :param options: Opciones adicionales para el reinicio.
        :param skip_countdown: Si es True, omite la cuenta atrás.
        :param visual: Si es True, inicializa la visualización de Pygame.
        :return: Observación inicial del entorno y un diccionario de información adicional.
        """
        super().reset(seed=seed)
        posiciones = [
            [5, 3], [6, 3], [7, 3], [8, 3], [9, 3],
            [5, 13], [6, 13], [7, 13], [8, 13], [9, 13]
        ]
        nombres_rojos = ["james_harden", "kawhi_leonard", "norman_powell", "kris_dunn", "ivica_zubac"]
        self.player_positions = [
            {
                "pos": pos.copy(),
                "color": board.BLUE if i < 5 else board.RED,
                "team": 1 if i < 5 else 2,
                "has_ball": False,
                "name": "defender_{i}" if i < 5 else nombres_rojos[i - 5]
            }
            for i, pos in enumerate(posiciones)
        ]
        self.current_player_idx = 5
        self.player_positions[self.current_player_idx]["has_ball"] = True
        self.ball_pos = self.player_positions[self.current_player_idx]["pos"].copy()
        self.score_red = 0
        self.score_blue = 0
        self.done = False
        self.last_shot_status = 'not_applicable'

        if visual and not skip_countdown:
            self._countdown()

        return self._get_obs(), {}

    def step(self, actions) -> tuple:
        """
        Realiza un paso en el entorno.
        :param actions: Lista de acciones a realizar por los jugadores.
        :return: Tupla (observación, recompensa, estado final, información adicional).
        """
        assert len(actions) == 5
        reward = 0
        self.last_shot_status = 'not_applicable'

        # Reasignar acciones no válidas a una aleatoria válida
        for i in range(5):
            player_idx = 5 + i
            valid = self._get_valid_actions(player_idx)
            if actions[i] not in valid:
                actions[i] = random.choice(valid)

        for i, action in enumerate(actions):
            player_idx = 5 + i
            player = self.player_positions[player_idx]
            r, c = player["pos"]

            if action == 4:
                if self._is_intercepted([r, c], intercept_prob=0.3):
                    self.last_shot_status = 'intercepted_shot'
                    reward = -1.0
                    self.done = True
                    return self._get_obs(), reward, self.done, False, self._build_info()
                else:
                    reward += self._calculate_shot_reward(r, c)

            elif action == 5:
                receivers = [j for j in range(5, 10) if j != player_idx]
                new_receiver = random.choice(receivers)
                new_pos = self.player_positions[new_receiver]["pos"]

                if self._is_intercepted(new_pos, intercept_prob=0.5):
                    self.last_shot_status = 'intercepted_pass'
                    reward = -1.0
                    self.done = True
                    return self._get_obs(), reward, self.done, False, self._build_info()
                
                self.player_positions[self.current_player_idx]["has_ball"] = False
                self.current_player_idx =new_receiver
                self.player_positions[self.current_player_idx]["has_ball"] = True
                self.ball_pos = new_pos

            elif action in [0, 1, 2, 3]:
                dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.rows and 0 <= new_c <= 13:
                    if not self._is_cell_occupied((new_r, new_c)) and board.get_cell_color(new_r, new_c) != 'BLACK':
                        player["pos"] = [new_r, new_c]
                        if player_idx == self.current_player_idx:
                            self.ball_pos = player["pos"].copy()

        # Movimiento defensivo azul: aleatorio ya que ahora no se controla con el agente 
        for i in range(5):
            r, c = self.player_positions[i]["pos"]
            random.shuffle(moves := [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]) # Pueden moverse en cualquier dirección o quedarse quietos
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc <= 13:
                    if not self._is_cell_occupied((nr, nc)) and board.get_cell_color(nr, nc) != 'BLACK':
                        self.player_positions[i]["pos"] = [nr, nc]
                        break

        return self._get_obs(), reward, self.done, False, self._build_info()

