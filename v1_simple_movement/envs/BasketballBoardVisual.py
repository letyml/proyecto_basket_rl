import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
import board

class BasketballBoardEnv(gym.Env):
    """
    Entorno de baloncesto para visualización en Pygame.
    Este entorno simula un tablero de baloncesto y permite a los jugadores moverse y lanzar el balón.
    """
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, config=None) -> None:
        '''
        Inicializa el entorno de baloncesto.
        :param render_mode: Modo de renderizado (por defecto "human").
        :param config: Configuración adicional (no utilizada en este caso).
        '''
        super().__init__()
        self.render_mode = render_mode

        self.rows = 15
        self.cols = 28
        self.action_space = spaces.Discrete(5)  # 0=↑, 1=↓, 2=←, 3=→, 4=lanzar

        # Restricción a mitad ofensiva: columnas 0–13
        low = np.array([0, 0, 0, 0])
        high = np.array([self.rows - 1, 13, self.rows - 1, 13])
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

        self.player_positions = [
            {"pos": [2, 7], "color": board.BLUE, "team": 1},
            {"pos": [4, 1], "color": board.BLUE, "team": 1},
            {"pos": [10, 3], "color": board.BLUE, "team": 1},
            {"pos": [7, 10], "color": board.BLUE, "team": 1},
            {"pos": [13, 5], "color": board.BLUE, "team": 1},
            {"pos": [2, 7], "color": board.RED, "team": 2},
            {"pos": [4, 9], "color": board.RED, "team": 2},
            {"pos": [10, 8], "color": board.RED, "team": 2},
            {"pos": [7, 12], "color": board.RED, "team": 2},
            {"pos": [13, 11], "color": board.RED, "team": 2},
        ]

        self.current_player_idx = 5
        self.ball_pos = self.player_positions[self.current_player_idx]["pos"].copy()
        self.score_blue = 0
        self.score_red = 0
        self.done = False

        pygame.init()
        self.screen = pygame.display.set_mode((board.WIDTH, board.HEIGHT))
        pygame.display.set_caption("Simulación Baloncesto 🏀")
        self.clock = pygame.time.Clock()
    
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
    
    def _countdown(self) -> None:
        '''
        Muestra una cuenta atrás de 5 segundos antes de iniciar el juego.
        '''
        screen = self.screen
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
            board.draw_court(screen)
            board.draw_players(screen, anim_players, current_player_idx=5)
            board.draw_ball_absolute(screen, [board.WIDTH // 2, (board.HEIGHT - 50) // 2])
            board.draw_score(screen, 0, 0)

            text = font.render(str(i), True, board.WHITE)
            screen.blit(text, (board.WIDTH // 2 - 30, board.HEIGHT // 2 - 80))
            pygame.display.flip()
            time.sleep(1)

    def reset(self, seed=None, options=None, skip_countdown=False) -> tuple:
        '''
        Reinicia el entorno y devuelve la observación inicial.
        :param seed: Semilla para la aleatoriedad (no utilizada en este caso).
        :param options: Opciones adicionales (no utilizadas en este caso).
        :return: Observación inicial y un diccionario vacío.
        '''
        super().reset(seed=seed)

        posiciones = [
            [5, 3], [6, 3], [7, 3], [8, 3], [9, 3],
            [5, 13], [6, 13], [7, 13], [8, 13], [9, 13]
        ]
        for i, pos in enumerate(posiciones):
            self.player_positions[i]["pos"] = pos.copy()

        self.current_player_idx = 5
        self.score_red = 0
        self.score_blue = 0
        self.ball_pos = self.player_positions[self.current_player_idx]["pos"].copy()
        self.done = False

        if not skip_countdown:
            self._countdown()

        return self._get_obs(), {}
    
    def step(self, action) -> tuple:
        '''
        Realiza un paso en el entorno con la acción dada.
        :param action: Acción a realizar (0=↑, 1=↓, 2=←, 3=→, 4=lanzar).
        :return: Tupla (observación, recompensa, terminado, truncado, información).
        '''
        reward = 0
        player_pos = self.player_positions[self.current_player_idx]["pos"]
        new_row, new_col = player_pos

        target_row, target_col = new_row, new_col

        if action == 0 and player_pos[0] > 0:
            target_row -= 1
        elif action == 1 and player_pos[0] < self.rows - 1:
            target_row += 1
        elif action == 2 and player_pos[1] > 0:
            target_col -= 1
        elif action == 3 and player_pos[1] < 13:
            target_col += 1
        elif action == 4:
            r, c = self.player_positions[self.current_player_idx]["pos"]
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

    def render(self) -> None:
        '''
        Renderiza el entorno en la pantalla.
        :return: None
        '''
        if self.render_mode != "human":
            return

        board.draw_court(self.screen)
        board.draw_players(self.screen, self.player_positions, self.current_player_idx)
        board.draw_ball_grid(self.screen, self.ball_pos)
        board.draw_score(self.screen, self.score_blue, self.score_red)

        if self.done:
            font = pygame.font.SysFont("Arial", 50, bold=True)
            text = font.render("¡Encestado!", True, board.WHITE)
            self.screen.blit(text, (board.WIDTH // 2 - 135, board.HEIGHT // 2 - 60))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self) -> None:
        '''
        Cierra el entorno y libera los recursos.
        :return: None
        '''
        pygame.quit()

if __name__ == "__main__":
    env = BasketballBoardEnv(render_mode="human")
    obs, info = env.reset()
    done = False

    env.render()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Acción: {action}, Recompensa: {reward}")
        env.render()

    time.sleep(2)
    print("\nFin del episodio!")
    env.close()