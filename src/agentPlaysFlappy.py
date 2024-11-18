import gymnasium as gym
from stable_baselines3 import PPO

# Modell und Umgebung laden
env = gym.make("FlappyBirdEnv-v0")
model = PPO.load("flappybird_ppo_model")

# Testen des Modells
obs = env.reset()
done = False
while not done:
    env.render()  # Zeigt die Umgebung an
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

env.close()
