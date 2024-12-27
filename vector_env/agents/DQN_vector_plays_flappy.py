#DQN_vector_plays_flappy.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from src.flappy import Flappy
import keyboard

# Wrappen im VecEnv
env = DummyVecEnv([lambda: Flappy(render_mode="human")])

# Gelerntes Modell laden
model = DQN.load(r"models/DQN/Flappy_Bird_DQN")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
