# 🏀 Basketball RL - Simulación de Estrategias con Aprendizaje Reforzado

Este proyecto implementa una simulación de estrategias de baloncesto en un entorno tipo juego de mesa. Utiliza Aprendizaje Reforzado mediante `Stable-Baselines3` y visualización con `pygame`. Cada versión incorpora mayor complejidad, tanto en el comportamiento del agente como en las reglas del entorno. El entrenamiento y seguimiento de métricas están gestionados con `MLflow`.

## 📁 Estructura del Proyecto

```
proyecto_basket_rl/
│
├── common/                                     # Código común reutilizable (callbacks, utilidades, tablero base, etc.)
│
├── mlruns/                                     # Registros de entrenamiento gestionados por MLflow
│
├── v1_simple_movement/                         # Versión 1: entorno básico, un solo jugador se mueve y lanza
│   ├── envs/
│   │   ├── BasketballBoardVisual.py            # Entorno con visualización Pygame
│   │   └── BasketballBoardCLI.py               # Entorno sin visualización (modo terminal)
│   ├── model/
│   │   ├── test_agent_v1.py                    # Visualización o testeo del agente entrenado       
│   │   └── train_agent_v1.py                   # Script principal de entrenamiento
│   ├── results/                                # Resultados del agente
|   |
│   ├── v1_simple_movement.txt                  # Descripción esquemática de esta versión
│
├── v2_pass_shoot_probability/                  # Versión 2: se añade lógica de pase y probabilidad de tiro
│   └── (estructura similar a v1)
│
├── v3_defense_interception/                    # Versión 3: mecánicas defensivas e intercepciones
│   └── (estructura similar a v1)
│
├── v4_player_roles_influence/                  # Versión 4: asignación de roles y datos de la NBA
|   ├── nba_api/                                # Extracción y preparación de los datos de la NBA
│   └── (estructura similar a v1)
│
├── analyze_version.ipynb                       # Notebook con análisis de los resultados de las versiones
│
├── launch_mlflow_ui.bat                        # Inicializa la interfaz web de MLflow en localhost:5000
│
├── README.md                                   # Este archivo: documentación general del proyecto
│
└── requirements.txt                            # Librerías necesarias para ejecutar el proyecto
```


## ⚙️ Instalación de Dependencias

```bash
pip install -r requirements.txt

```

## 🚀 Entrenamiento (ejemplo usando v1)

```bash
cd v1_simple_movement/model
python train_agent_v1.py
```

El modelo se guardará en `v1_simple_movement/results/model_v1_YYYYMMDD_HHMMSS/train_v1.zip`. En la carpeta `mlruns/` se almacenarán todos los registros del experimento.

### ▶️ Visualizar un Agente Entrenado

```bash
python test_agent_v1.py

```

Se solicitará el identificador del modelo (model_v1_YYYYMMDD_HHMMSS) y el número de episodios a visualizar.

> Este script usa `pygame` para mostrar cómo juega el agente en un episodio concreto.

### 📊 Visualizar Resultados en MLflow

Ejecuta el siguiente script:

```bash
launch_mlflow_ui.bat
```
A continuación, abre tu navegador en la dirección: `http://localhost:5000`
Desde ahí podrás explorar métricas, gráficas y detalles del entrenamiento.

## 📌 Créditos

Proyecto desarrollado por **Leticia Martínez Limeres** para el Trabajo de Fin de Máster del *Máster Universitario en Tecnologías de Análisis de Datos Masivos: Big Data* – Universidade de Santiago de Compostela (USC).
