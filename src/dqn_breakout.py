import gymnasium as gym

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3 import DQN

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # zufällige Aktion
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        observation, info = env.reset()
env.close()

