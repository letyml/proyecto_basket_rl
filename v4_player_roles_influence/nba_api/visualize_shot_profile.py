import pygame
import json
import os
import sys

# Añadir la ruta al módulo del tablero
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common')))
from board import WIDTH, HEIGHT, ROWS, COLS, CELL_WIDTH, CELL_HEIGHT, draw_court

# ========================================================
# 🔄 Elegir entre datos originales o completados (KNN)
# ========================================================
modo = input("¿Qué datos quieres visualizar? (o = originales, c = completos con KNN): ").strip().lower()
if modo == "c":
    json_filename = "player_shooting_profiles_knn.json"
else:
    json_filename = "player_shooting_profiles.json"

json_path = os.path.join(os.path.dirname(__file__), "data", json_filename)
with open(json_path, "r") as f:
    profiles = json.load(f)

# Lista de jugadores disponibles
PLAYER_LIST = list(profiles.keys())
player_index = 0

# ========================================================
# Inicializar pygame solo después de tener los datos
# ========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mapa de tiro sobre el tablero 🏀")
font = pygame.font.SysFont("Arial", 12, bold=True)

def get_color(accuracy) -> tuple:
    """
    Devuelve un color RGB basado en la precisión del tiro.
    :param accuracy: Precisión del tiro (0.0 a 1.0).
    :return: Tupla RGB representando el color.
    """
    r = int(180 * (1 - accuracy))
    g = int(180 * accuracy)
    return (r, g, 0)

def draw_profile(player_id) -> None:
    """
    Dibuja el perfil de tiro del jugador seleccionado en la pantalla.
    :param player_id: ID del jugador cuyo perfil se va a dibujar.
    """
    screen.fill((255, 255, 255))
    draw_court(screen)
    profile = profiles.get(player_id, {})

    for row in range(ROWS):
        for col in range(14):  # solo mitad ofensiva
            key = str((row, col))
            x = col * CELL_WIDTH + CELL_WIDTH // 2
            y = row * CELL_HEIGHT + CELL_HEIGHT // 2

            if key in profile:
                prob = profile[key].get("prob", 0.1)
                color = get_color(prob)
                text = font.render(f"{prob:.2f}", True, color)
            else:
                text = font.render("-", True, (160, 160, 160))

            screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

    label = font.render(f"{player_id.replace('_', ' ').title()}", True, (0, 0, 0))
    screen.blit(label, (WIDTH - label.get_width() - 10, 10))

    pygame.display.flip()

def main():
    global player_index
    running = True
    draw_profile(PLAYER_LIST[player_index])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    player_index = (player_index + 1) % len(PLAYER_LIST)
                    draw_profile(PLAYER_LIST[player_index])
                elif event.key == pygame.K_LEFT:
                    player_index = (player_index - 1) % len(PLAYER_LIST)
                    draw_profile(PLAYER_LIST[player_index])

    pygame.quit()

if __name__ == "__main__":
    main()
