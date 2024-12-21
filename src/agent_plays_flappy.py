from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from envs.environment_image import FlappyBirdEnv
import keyboard

# Umgebung einpacken
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="human")])

# Modell laden

model = PPO.load("D30_L0002_P3_600k_H15")

try:
    while True:  # Äußere Schleife für kontinuierliches Spielen
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

            if keyboard.is_pressed('q'):  # Prüfen, ob 'q' gedrückt wurde
                print("Spiel manuell beendet.")
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Programm beendet.")

finally:
    env.close()
