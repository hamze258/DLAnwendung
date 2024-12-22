from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.vector_env.flappy_vector_env import FlappyVectorEnv

# 1. Umgebung initialisieren und überprüfen
env = FlappyVectorEnv()
check_env(env, warn=True)

# 2. Trainingsumgebung erzeugen
vec_env = DummyVecEnv([lambda: FlappyVectorEnv()])

# 3. Evaluationsumgebung erzeugen
eval_env = DummyVecEnv([lambda: FlappyVectorEnv()])

# 4. Callbacks
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="models/DQN/",
    log_path="logs/DQN/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000, 
    save_path="checkpoints/DQN/", 
    name_prefix="flappy_bird_vector"
)

# 5. DQN-Modell initialisieren
model = DQN(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=0.0001,        # typischer LR-Wert für DQN
    buffer_size=100000,          # Replayspeicher
    learning_starts=10000,       # Erst nach 10k Steps aktiv lernen
    batch_size=32,               # Batch-Größe für das Training
    tau=0.8,                     # Polyak-Koeffizient (weiches Update)
    gamma=0.99,                  # Diskontierungsfaktor
    train_freq=4,                # Alle 4 Steps wird trainiert
    target_update_interval=10000,# Alle 10k Steps Update des Target-Netzwerks
    exploration_fraction=0.1,    # Prozentsatz der Trainingsschritte für epsilon decay
    exploration_final_eps=0.01,  # Minimales Epsilon (Exploration)
    tensorboard_log="tensorboard/DQN",
)

# 6. Training
model.learn(
    total_timesteps=3000000,
    callback=[eval_callback, checkpoint_callback],
)

# 7. Speichern
model.save("dqn_flappy_bird_final")
