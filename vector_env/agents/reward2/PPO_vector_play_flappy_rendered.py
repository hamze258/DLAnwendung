
import os
import time
import imageio

#DQN_vector_plays_flappy.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from vector_env.agents.reward2.flappy_vector_env import FlappyBirdEnv
import time

# Videoausgabeordner erstellen
video_dir = r"vector_env\videos"
os.makedirs(video_dir, exist_ok=True)


model = PPO.load(r"vector_env\models\PPO\training1\best_model")


# Name des Videos
video_path = os.path.join(video_dir, "PPO_1.mp4")

# Umgebung erstellen mit "rgb_array" Render-Modus
#env = FlappyBirdEnv(render_mode="rgb_array")

env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])


# Anzahl der Episoden, die Sie aufzeichnen möchten
num_episodes = 15  # Ändern Sie dies auf die gewünschte Anzahl

# Video Writer mit imageio
fps = 30  # Frames pro Sekunde

with imageio.get_writer(video_path, fps=fps) as video:
    for episode in range(num_episodes):
        #try:
        obs = env.reset()
        done = False

        while not done:
            # Render frame als RGB-Array
            frame = env.envs[0].render()
            video.append_data(frame)

            action, _ = model.predict(obs, deterministic=True)

            # Aktion ausführen
            obs, reward, done, info = env.step(action)
            #time.sleep(0.05)  # Optional: für eine flüssigere Darstellung



env.close()

print(f"Video wurde in {video_path} gespeichert.")
