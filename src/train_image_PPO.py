from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.environment_image import FlappyBirdEnv

# Initialisiere die Umgebung
env = FlappyBirdEnv()
check_env(env, warn=True)

# Trainingsumgebung
vec_env = DummyVecEnv([lambda: FlappyBirdEnv()])
vec_env = VecTransposeImage(vec_env)

# Evaluationsumgebung
eval_env = DummyVecEnv([lambda: FlappyBirdEnv()])
eval_env = VecTransposeImage(eval_env)

# Callbacks
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./ppo_flappy_bird/",
    log_path="./ppo_flappy_bird/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000, save_path="./ppo_checkpoints/", name_prefix="flappy_bird"
)

# Modellinitialisierung
model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=4096,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    tensorboard_log="./ppo_flappy_bird_tensorboard/",
)

# Training
model.learn(
    total_timesteps=600000,
    callback=[eval_callback, checkpoint_callback],
)

# Speichern
model.save("ppo_flappy_bird_final")
