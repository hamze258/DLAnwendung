import gym
import numpy as np
import random

# Importiere die FlappyBirdEnv-Klasse
from envs.environment import FlappyBirdEnv

def run_baseline_agent(env, episodes=10):
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Entpacke die Beobachtungen
            bird_y = obs[0]
            bird_velocity = obs[1]
            next_pipe_x = obs[2]
            next_pipe_y = obs[3]

            # Agentenlogik
            if bird_y > next_pipe_y or next_pipe_x < 150:
                action = 1  # Flap
            else:
                action = 0  # Nicht-Flap

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {episode + 1}: Gesamtbelohnung = {total_reward}")
    env.close()


if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode="human")
    run_baseline_agent(env, episodes=10)

