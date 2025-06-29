#!/usr/bin/python
# -*- coding: utf-8 -*-

import pygame

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

# Definición de colores
WHITE = (255, 255, 255)
BLACK = (76, 76, 76)
RED = (224, 16, 3)
BLUE = (13, 23, 218)
ORANGE = (247, 188, 109)
DARK_ORANGE = (232, 147, 35)
GREY = (230, 230, 230)
CIAN = (3, 211, 224)

# Definición de constantes
WIDTH, HEIGHT = 900, 590
ROWS, COLS = 15, 28 # 15 filas y 28 columnas 
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = (HEIGHT - 50) // ROWS  
clock = pygame.time.Clock()
FPS = 60
player_radius = 20

# ==============================
# FUNCIONES DE DIBUJO
# ==============================

def draw_court(screen) -> None:
    """
    Dibuja la cancha de baloncesto en la pantalla.
    :param screen: Pantalla de Pygame donde se dibuja la cancha.
    :return: None
    """
    screen.fill(GREY)
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            if row == 7 and (col == 0 or col == COLS - 1):
                pygame.draw.rect(screen, BLACK, rect)
            elif row in range(5, 10) and (col < 5 or col > 22):
                pygame.draw.rect(screen, DARK_ORANGE, rect)
            elif (
                (row == 1 and (col < 3 or col > 24)) or
                (row == 2 and (col < 5 or col > 22)) or
                (row == 3 and (col < 6 or col > 21)) or
                (row == 4 and (col < 7 or col > 20)) or
                (row == 5 and (col < 8 or col > 19)) or
                (row == 6 and (col < 8 or col > 19)) or
                (row == 8 and (col < 8 or col > 19)) or
                (row == 9 and (col < 8 or col > 19)) or
                (row == 10 and (col < 7 or col > 20)) or
                (row == 11 and (col < 6 or col > 21)) or
                (row == 12 and (col < 5 or col > 22)) or
                (row == 13 and (col < 3 or col > 24)) or
                (row == 7 and (col == 7 or col == 20)) or
                (row in range(6, 9) and (col < 7 or col > 20))
            ):
                pygame.draw.rect(screen, ORANGE, rect)
            else:
                pygame.draw.rect(screen, GREY, rect)

            pygame.draw.rect(screen, WHITE, rect, 1)

    pygame.draw.line(screen, BLACK, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT - 50), 2)
    center = (WIDTH // 2, (HEIGHT - 50) // 2)
    pygame.draw.circle(screen, BLACK, center, 50, 2)
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, WIDTH // 2, HEIGHT - 50), 3)
    pygame.draw.rect(screen, BLACK, pygame.Rect(WIDTH // 2, 0, WIDTH // 2, HEIGHT - 50), 3)

def draw_players(screen, player_positions, current_player_idx=None) -> None:
    """
    Dibuja los jugadores en la cancha.
    :param screen: Pantalla de Pygame donde se dibujan los jugadores.
    :param player_positions: Lista de posiciones de los jugadores.
    :param current_player_idx: Índice del jugador actual (opcional).
    :return: None
    """
    for i, player in enumerate(player_positions):
        row, col = player["pos"]
        x = col * CELL_WIDTH + CELL_WIDTH // 2
        y = row * CELL_HEIGHT + CELL_HEIGHT // 2
        scale = 0.5 if i == current_player_idx else 0.5
        pygame.draw.circle(screen, player["color"], (x, y), int(player_radius * scale))

def draw_ball_absolute(screen, ball_pos_absolute) -> None:
    """
    Dibuja el balón usando coordenadas absolutas en píxeles (x, y).
    :param screen: Pantalla de Pygame donde se dibuja el balón.
    :param ball_pos_absolute: Posición del balón en coordenadas absolutas (x, y).
    :return: None
    """
    x, y = ball_pos_absolute
    pygame.draw.circle(screen, CIAN, (x, y), 10, 0)
    pygame.draw.circle(screen, GREY, (x, y), 10, 2)

def draw_ball_grid(screen, ball_pos_grid, radius=7) -> None:
    """
    Dibuja el balón en coordenadas de celda (fila, columna) con un radio más pequeño.
    :param screen: Pantalla de Pygame donde se dibuja el balón.
    :param ball_pos_grid: Posición del balón en coordenadas de celda (fila, columna).
    :param radius: Radio del balón.
    :return: None
    """
    row, col = ball_pos_grid
    x = col * CELL_WIDTH + CELL_WIDTH // 2
    y = row * CELL_HEIGHT + CELL_HEIGHT // 2
    pygame.draw.circle(screen, CIAN, (x, y), radius, 0)
    pygame.draw.circle(screen, GREY, (x, y), radius, 2)

def draw_score(screen, score_blue, score_red) -> None:
    """
    Dibuja el marcador en la parte inferior de la pantalla.
    :param screen: Pantalla de Pygame donde se dibuja el marcador.
    :param score_blue: Puntuación del equipo azul.
    :param score_red: Puntuación del equipo rojo.
    :return: None
    """
    font = pygame.font.SysFont("Arial", 35, bold=True)
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, HEIGHT - 50, WIDTH, 50))
    blue_text = font.render(str(score_blue), True, GREY)
    red_text = font.render(str(score_red), True, GREY)
    screen.blit(blue_text, (WIDTH // 2 - blue_text.get_width() - 30, HEIGHT - 48))
    screen.blit(red_text, (WIDTH // 2 + 30, HEIGHT - 48))

def get_cell_color(row, col) -> str:
    """
    Devuelve el color de la celda en función de su posición (fila, columna).
    :param row: Fila de la celda.
    :param col: Columna de la celda.
    :return: Color de la celda.
    """
    if row == 7 and (col == 0 or col == COLS - 1):
        return 'BLACK'
    elif row in range(5, 10) and (col < 5 or col > 22):
        return 'DARK_ORANGE'
    elif (
        (row == 1 and (col < 3 or col > 24)) or
        (row == 2 and (col < 5 or col > 22)) or
        (row == 3 and (col < 6 or col > 21)) or
        (row == 4 and (col < 7 or col > 20)) or
        (row == 5 and (col < 8 or col > 19)) or
        (row == 6 and (col < 8 or col > 19)) or
        (row == 8 and (col < 8 or col > 19)) or
        (row == 9 and (col < 8 or col > 19)) or
        (row == 10 and (col < 7 or col > 20)) or
        (row == 11 and (col < 6 or col > 21)) or
        (row == 12 and (col < 5 or col > 22)) or
        (row == 13 and (col < 3 or col > 24)) or
        (row == 7 and (col == 7 or col == 20)) or
        (row in range(6, 9) and (col < 7 or col > 20))
    ):
        return 'ORANGE'
    else:
        return 'GREY'

def get_zone_score(row, col) -> int:
    """
    Devuelve la puntuación de la zona en función de su posición (fila, columna).
    :param row: Fila de la celda.
    :param col: Columna de la celda.
    :return: Puntuación de la zona.
    """
    color = get_cell_color(row, col)
    if color == 'GREY':
        return 3
    elif color in ['ORANGE', 'DARK_ORANGE']:
        return 2
    else:
        return 0

# ==============================
# MAIN PARA VER TABLERO SOLO
# ==============================

def main() -> None:
    """
    Función principal para probar la simulación del tablero de baloncesto.
    :return: None
    """

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tablero de Baloncesto 🏀")

    player_positions = [
        {"pos": [2, 7], "color": BLUE},
        {"pos": [4, 1], "color": BLUE},
        {"pos": [10, 3], "color": BLUE},
        {"pos": [7, 10], "color": BLUE},
        {"pos": [13, 5], "color": BLUE},
        {"pos": [2, 20], "color": RED},
        {"pos": [4, 26], "color": RED},
        {"pos": [10, 24], "color": RED},
        {"pos": [7, 17], "color": RED},
        {"pos": [13, 22], "color": RED},
    ]
    ball_pos_absolute = [WIDTH // 2, (HEIGHT - 50) // 2]
    score_blue = 0
    score_red = 0

    running = True
    while running:
        draw_court(screen)
        draw_players(screen, player_positions)
        draw_ball_absolute(screen, ball_pos_absolute)
        draw_score(screen, score_blue, score_red)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(FPS)

    pygame.quit()
    print("Simulación finalizada.")

if __name__ == "__main__":
    main()
