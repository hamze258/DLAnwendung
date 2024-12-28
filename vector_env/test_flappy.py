import random
from agents.flappy_vector_env import FlappyBirdEnv
import time
env = FlappyBirdEnv(render_mode="human")  # Oder render_mode=None für Training

def run_baseline(env,episodes=10):
    for i in range(episodes):
        obs,_ = env.reset()
        done = False
        while not done:
            
            bird_y=obs[0]
            bird_velocity=obs[1]
            next_pipe_x=obs[2]
            next_pipe_top_y=obs[3]
            next_pipe_bottom_y=obs[4]
            
            if bird_y < next_pipe_top_y:
                action = 1
            else:
                action = 0
            
            obs, reward, done, _, info = env.step(action)
            print("Belohnung:", reward, "Score:", info["score"])
            env.render()  # Spiel rendern

    env.close()

run_baseline(env,episodes=10)
