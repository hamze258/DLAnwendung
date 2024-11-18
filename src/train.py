import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure



gym.register("FlappyBirdEnv-v0", entry_point="environment:FlappyBirdEnv")

# 1. Umgebung initialisieren
env = gym.make("FlappyBirdEnv-v0")  # Stelle sicher, dass "FlappyBird-v0" richtig installiert ist

# 2. Modell initialisieren
# Verwende das PPO Modell mit Standard-Hyperparametern oder eigenen Anpassungen
model = PPO("MlpPolicy", env, verbose=1)

# 3. Checkpoint Callback, um das Modell regelmäßig zu speichern
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='flappybird_ppo')

# 4. Training starten
# Passe die Timesteps je nach Anforderungen an
model.learn(total_timesteps=1000000, callback=checkpoint_callback)

# 5. Modell speichern
model.save("flappybird_ppo_model")

# Umgebung schließen
env.close()

print("Training abgeschlossen und Modell gespeichert!")
