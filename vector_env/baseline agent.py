# from agents.flappy_vector_env import FlappyBirdEnv
# from gymnasium.wrappers import RecordVideo
# import os
# import time


# # Environment erstellen
# env = FlappyBirdEnv(render_mode="human")

# # Baseline-Agent
# def run_baseline(env,episodes=10):
#     for i in range(episodes):
#         obs,_ = env.reset()
#         done = False
#         while not done:
            
#             bird_y=obs[0]
#             bird_velocity=obs[1]
#             next_pipe_x=obs[2]
#             next_pipe_top_y=obs[3]
#             next_pipe_bottom_y=obs[4]
            
#             if bird_y < next_pipe_top_y:
#                 action = 1
#             else:
#                 action = 0
            
#             obs, reward, done, _, info = env.step(action)
#             time.sleep(0.05)
#             print("Belohnung:", reward, "Score:", info["score"])
#             env.render()  # Spiel rendern

#     env.close()

# run_baseline(env,episodes=2)
from agents.flappy_vector_env import FlappyBirdEnv
import os
import time
import imageio

# Videoausgabeordner erstellen
video_dir = "./videos"
os.makedirs(video_dir, exist_ok=True)

# Name des Videos
video_path = os.path.join(video_dir, "kein_baseline_mehr.mp4")

# Umgebung erstellen mit "rgb_array" Render-Modus
env = FlappyBirdEnv(render_mode="rgb_array")

# Anzahl der Episoden, die Sie aufzeichnen möchten
num_episodes = 2  # Ändern Sie dies auf die gewünschte Anzahl

# Video Writer mit imageio
fps = 20  # Frames pro Sekunde
with imageio.get_writer(video_path, fps=fps) as video:
    for episode in range(num_episodes):
        try:
            obs, _ = env.reset()
            done = False

            while not done:
                # Render frame als RGB-Array
                frame = env.render()
                video.append_data(frame)

                bird_y = obs[0]
                bird_velocity = obs[1]
                next_pipe_x = obs[2]
                next_pipe_top_y = obs[3]
                next_pipe_bottom_y = obs[4]

                # Einfache Regel: Flappen, wenn der Vogel unter dem oberen Rohr ist
                if bird_y < next_pipe_top_y:
                    action = 1  # Flappen
                else:
                    action = 0  # Nichts tun

                # Aktion ausführen
                obs, reward, done, _, info = env.step(action)
                time.sleep(0.05)  # Optional: für eine flüssigere Darstellung

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

env.close()

print(f"Video wurde in {video_path} gespeichert.")
