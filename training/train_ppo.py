from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Stelle sicher, dass du die FlappyBirdEnv-Klasse vorher definiert hast.
from envs.image_env.flappy_image_env import FlappyBirdEnv  # Ersetze dies durch deinen Pfad, wenn die Env in einer anderen Datei liegt.

# Initialisiere die Umgebung
env = FlappyBirdEnv()

# Überprüfe die Kompatibilität der Umgebung mit Stable Baselines3
check_env(env, warn=True)

# Verwende einen Vektorisierer (für bessere Parallelisierung, auch mit nur 1 Env)
vec_env = DummyVecEnv([lambda: FlappyBirdEnv()])

# Callbacks zur besseren Trainingskontrolle
eval_callback = EvalCallback(
    vec_env,
    best_model_save_path="models/PPO/",
    log_path="logs/PPO/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000, save_path="checkpoints/PPO", name_prefix="flappy_bird"
)

# Initialisiere das PPO-Modell mit der Umgebung
model = PPO(
    "MlpPolicy",  # Verwende ein Multi-Layer-Perceptron (MLP) als Policy
    vec_env,
    verbose=1,
    learning_rate=0.0002,  # Standard-Learning-Rate für PPO
    n_steps=2048,  # Anzahl der Schritte pro Batch
    batch_size=64,  # Minibatchgröße
    gae_lambda=0.95,  # GAE Glättungsfaktor
    gamma=0.99,  # Discount-Faktor
    clip_range=0.2,  # Clipping-Faktor für PPO-Updates
    tensorboard_log="tensorboard/PPO",  # Tensorboard-Logs
)

# Starte das Training
model.learn(
    total_timesteps=1000000,  # Passe diese Zahl an deine Ressourcen an
    callback=[eval_callback, checkpoint_callback]
)

# Speichere das trainierte Modell
model.save("ppo_flappy_bird_final")