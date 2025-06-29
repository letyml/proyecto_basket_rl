from BasketballBoardBaseEnv import BasketballBoardBaseEnvV3
import random

class BasketballBoardCLIV3(BasketballBoardBaseEnvV3):
    """
    Clase para la simulación CLI del entorno de baloncesto.
    Esta clase hereda de BasketballBoardBaseEnvV3 y proporciona una interfaz de línea de comandos
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
    env = BasketballBoardCLIV3()
    obs, _ = env.reset()
    done = False
    step = 0

    print("Simulación CLI iniciada con acciones aleatorias.")

    while not done:
        actions = [random.randint(0, 5) for _ in range(5)]
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        step += 1

        base, final, penalty, count, defenders = info["prob_info"]

        print(f"Paso {step:02d} | Acciones: {actions} | Recompensa: {reward:.2f} | "
            f"Jugador #{info['player_idx']} en {info['player_position']} | "
            f"Balón: {info['ball_position']} | ProbBase: {base:.2f} | ProbFinal: {final:.2f} | "
            f"Penalización: {penalty:.2f} | Num defensores cercanos: {count} | Ids defensores: {defenders} | "
            f"Estado tiro: {info['shot']} | Score: 🔵 {info['score_blue']} - 🔴 {info['score_red']} | todos: {info['all_players']}\n")
        
    print(f"\nEpisodio finalizado en {step} pasos.")
