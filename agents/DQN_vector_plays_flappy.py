#DQN_vector_plays_flappy.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from envs.vector_env.flappy_vector_env import FlappyVectorEnv
import keyboard

# Wrappen im VecEnv
env = DummyVecEnv([lambda: FlappyVectorEnv(render_mode="human")])

# Gelerntes Modell laden
model = DQN.load(r"models\DQN\best_model.zip")

try:
    while True:
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

            if keyboard.is_pressed('q'):
                print("Manuell abgebrochen.")
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Programm beendet.")
finally:
    env.close()
