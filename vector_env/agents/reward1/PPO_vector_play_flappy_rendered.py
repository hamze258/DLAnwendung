import os
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from vector_env.agents.reward1.flappy_vector_env import FlappyBirdEnv

# Videoausgabeordner erstellen
video_dir = r"vector_env\videos"
os.makedirs(video_dir, exist_ok=True)

# Modell laden
model = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")

# Pfad für das Video festlegen
video_path = os.path.join(video_dir, "PPO1.mp4")

# Umgebung mit "rgb_array" Render-Modus erstellen
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

# Anzahl der zu zeichnenden Episoden
num_episodes = 15

# Video Writer initialisieren
fps = 30
with imageio.get_writer(video_path, fps=fps) as video:
    for episode in range(num_episodes):
        obs = env.reset()
        done = [False]  # DummyVecEnv gibt Listen zurück
        
        while not done[0]:
            # Frame rendern und hinzufügen
            frame = env.envs[0].render()
            video.append_data(frame)
            
            # Aktion vom Modell vorhersagen
            action, _ = model.predict(obs, deterministic=True)
            
            # Aktion ausführen
            obs, reward, done, info = env.step(action)

# Umgebung schließen
env.close()

print(f"Video wurde in {video_path} gespeichert.")
