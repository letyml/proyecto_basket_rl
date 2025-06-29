import os
import sys
import pygame
from BasketballBoardBaseEnv import BasketballBoardBaseEnvV3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
import board

class BasketballBoardVisualV3(BasketballBoardBaseEnvV3):
    """
    Clase para la simulación visual del entorno de baloncesto.
    Esta clase hereda de BasketballBoardBaseEnvV3 y proporciona una interfaz gráfica
    utilizando Pygame para visualizar el entorno de baloncesto.
    """
    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(self, render_mode="human") -> None:
        """
        Inicializa el entorno de baloncesto.
        :param render_mode: Modo de renderizado (por defecto "human").
        """
        self.render_mode = render_mode
        pygame.init()
        self.screen = pygame.display.set_mode((board.WIDTH, board.HEIGHT))
        pygame.display.set_caption("Simulación Baloncesto")
        self.clock = pygame.time.Clock()

        super().__init__()

    def render(self) -> None:
        """
        Renderiza el entorno de baloncesto.
        Dibuja la cancha, los jugadores, el balón y el marcador en la pantalla.
        """
        if self.render_mode != "human":
            return

        board.draw_court(self.screen)
        board.draw_players(self.screen, self.player_positions, self.current_player_idx)
        board.draw_ball_grid(self.screen, self.ball_pos)
        board.draw_score(self.screen, self.score_blue, self.score_red)

        if self.done and self.last_shot_status in ['shot_successful', 'shot_failed']:
            font = pygame.font.SysFont("Arial", 50, bold=True)
            mensaje = "¡Encestado!" if self.last_shot_status == 'shot_successful' else "Fallido..."
            text = font.render(mensaje, True, board.WHITE)
            text_rect = text.get_rect(center=(board.WIDTH // 2, board.HEIGHT // 2 - 30))
            self.screen.blit(text, text_rect)

        elif self.done and self.last_shot_status in ['intercepted_shot', 'intercepted_pass']:
            font = pygame.font.SysFont("Arial", 50, bold=True)
            mensaje = "Pelota Perdida"
            text = font.render(mensaje, True, board.WHITE)
            text_rect = text.get_rect(center=(board.WIDTH // 2 , board.HEIGHT // 2 - 30))
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(1.5)


    def close(self) -> None:
        """
        Cierra el entorno de baloncesto y libera los recursos utilizados.
        """
        pygame.quit()

if __name__ == "__main__":
    env = BasketballBoardVisualV3()
    obs, _ = env.reset(skip_countdown=False, visual=True)
    done = False
    step = 0
    print("Simulación visual iniciada (modo aleatorio)\n")

    env.render()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        acciones = [int(env.action_space.sample()[i]) for i in range(5)]
        obs, reward, terminated, truncated, info = env.step(acciones)
        done = terminated or truncated
        step += 1

        base, final, penalty, count, defenders = info["prob_info"]

        print(f"Paso {step:02d} | Acciones: {acciones} | Recompensa: {reward:.2f} | "
            f"Jugador #{info['player_idx']} en {info['player_position']} | "
            f"Balón: {info['ball_position']} | ProbBase: {base:.2f} | ProbFinal: {final:.2f} | "
            f"Penalización: {penalty:.2f} | Num defensores cercanos: {count} | Ids defensores: {defenders} | "
            f"Estado tiro: {info['shot']} | Score: 🔵 {info['score_blue']} - 🔴 {info['score_red']} | todos: {info['all_players']}")

        env.render()

