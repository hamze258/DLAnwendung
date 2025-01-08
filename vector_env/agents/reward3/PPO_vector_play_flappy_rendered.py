
import os
import imageio

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from vector_env.agents.reward3.flappy_vector_env import FlappyBirdEnv

# Videoausgabeordner erstellen
video_dir = r"vector_env\videos\reward3"
os.makedirs(video_dir, exist_ok=True)


model = PPO.load(r"vector_env\models\PPO\training16\best_model.zip")


# Name des Videos
video_path = os.path.join(video_dir, "PPO_3.mp4")


env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

num_episodes = 1

fps = 60  # Frames pro Sekunde

with imageio.get_writer(video_path, fps=fps) as video:
    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        for i in range(1000):
            # Render frame als RGB-Array
            frame = env.envs[0].render()
            video.append_data(frame)

            action, _ = model.predict(obs, deterministic=True)

            # Aktion ausführen
            obs, reward, done, info = env.step(action)
            print(info)

env.close()

print(f"Video wurde in {video_path} gespeichert.")
