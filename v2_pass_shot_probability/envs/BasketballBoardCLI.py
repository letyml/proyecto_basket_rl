from BasketballBoardBaseEnv import BasketballBoardBaseEnvV2
import random

class BasketballBoardCLIV2(BasketballBoardBaseEnvV2):
    """
    Clase para la simulación CLI del entorno de baloncesto.
    Esta clase hereda de BasketballBoardBaseEnvV2 y proporciona una interfaz de línea de comandos
    para interactuar con el entorno de baloncesto.
    """
    def reset(self, **kwargs) -> tuple:
        """
        Reinicia el entorno y devuelve la observación inicial.
        :param kwargs: Opciones adicionales (no utilizadas en este caso).
        :return: Observación inicial y un diccionario vacío.
        """
        # Forzar visual=False para que nunca se llame a _countdown()
        return super().reset(visual=False, **kwargs)

    def render(self): 
        pass # No se necesita renderizar en CLI

    def close(self):
        pass # No se necesita cerrar nada en CLI


if __name__ == "__main__":
    env = BasketballBoardCLIV2()
    obs, _ = env.reset()
    done = False
    step = 0

    print("Simulación CLI iniciada con acciones aleatorias.")

    while not done:
        actions = [random.randint(0, 5) for _ in range(5)]
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        step += 1

        print(f"[Paso {step}] Acciones: {actions} | Recompensa: {reward:.2f} | "
              f"Jugador #{info['player_idx']} en {info['player_position']} | "
              f"Balón: {info['ball_position']} | Probabilidad: {info['probability']:.2f} | "
              f"Tiro: {info['shot']}")
        
    print(f"\nEpisodio finalizado en {step} pasos.")
    print(f"🔵 Azul: {info['score_blue']}  🔴 Rojo: {info['score_red']}")
