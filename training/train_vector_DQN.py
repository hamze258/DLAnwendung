from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch

from envs.vector_env.flappy_vector_env import FlappyVectorEnv

# 1. Umgebung initialisieren und überprüfen
env = FlappyVectorEnv()
check_env(env, warn=True)

# 2. Trainingsumgebung erzeugen
vec_env = DummyVecEnv([lambda: FlappyVectorEnv()])

# 3. Evaluationsumgebung erzeugen
eval_env = DummyVecEnv([lambda: FlappyVectorEnv()])


# 5. Callbacks
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="models/DQN/",
    log_path="logs/DQN/",
    eval_freq=10000,  # Erhöht die Evaluationshäufigkeit für besseres Monitoring
    n_eval_episodes=10,  # Mehr Episoden für aussagekräftige Evaluation
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,  # Weniger häufig speichern, um Platz zu sparen
    save_path="checkpoints/DQN/", 
    name_prefix="flappy_bird_vector"
)

# 6. Dynamische GPU- oder CPU-Auswahl
device = "cuda" if torch.cuda.is_available() else "cpu"

# 7. Hyperparameter für DQN optimieren
model = DQN(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=1e-4,  # Typischer Wert, bereits passend
    buffer_size=200000,  # Erhöhter Replay-Speicher für komplexeres Lernen
    learning_starts=10000,
    batch_size=64,  # Größere Batches für stabileres Training
    tau=1.0,  # Standardwert für weiches Update, passt zu DQN
    gamma=0.99,  # Diskontierungsfaktor bleibt gleich
    train_freq=(1, "step"),  # Trainiert nach jedem Schritt
    gradient_steps=1,  # Aktualisiert das Netzwerk nach jedem Sample
    target_update_interval=5000,  # Häufigeres Update des Zielnetzwerks
    exploration_fraction=0.2,  # Langsamerer Epsilon-Abbau
    exploration_final_eps=0.02,  # Minimales Epsilon leicht erhöht
    tensorboard_log="tensorboard/DQN",
    device=device,
)



# 9. Training mit optimierten Parametern
model.learn(
    total_timesteps=3000000,
    callback=[eval_callback, checkpoint_callback],
    log_interval=1000,  # Häufigeres Logging
    progress_bar=True,  # Fortschrittsbalken für bessere Übersicht
)

# 10. Modell speichern
model.save("dqn_flappy_bird_final_optimized")
