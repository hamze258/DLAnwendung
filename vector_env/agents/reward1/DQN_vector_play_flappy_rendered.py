import os
import imageio

#DQN_vector_plays_flappy.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from vector_env.agents.reward1.flappy_vector_env import FlappyBirdEnv
import time

# Videoausgabeordner erstellen
video_dir = r"vector_env\videos\reward1"
os.makedirs(video_dir, exist_ok=True)


model = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")


# Name des Videos
video_path = os.path.join(video_dir, "DQN1.mp4")

# Umgebung erstellen mit "rgb_array" Render-Modus
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

# Anzahl der Episoden
num_episodes = 15 

fps = 30  # Frames pro Sekunde

with imageio.get_writer(video_path, fps=fps) as video:
    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            # Render frame als RGB-Array
            frame = env.envs[0].render()
            video.append_data(frame)

            action, _ = model.predict(obs, deterministic=True)

            # Aktion ausführen
            obs, reward, done, info = env.step(action)
env.close()

print(f"Video wurde in {video_path} gespeichert.")
