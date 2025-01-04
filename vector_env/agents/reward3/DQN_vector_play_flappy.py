#DQN_vector_plays_flappy.py
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from vector_env.agents.flappy_vector_env import FlappyBirdEnv
import keyboard

# Wrappen im VecEnv
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

# Gelerntes Modell laden
model = DQN.load(r"vector_env\models\DQN\training4\best_model.zip")

try:
    for i in range(100):  # Äußere Schleife für kontinuierliches Spielen
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            #time.sleep(0.02)
            env.render()

            if keyboard.is_pressed('q'):  # Prüfen, ob 'q' gedrückt wurde
                print("Spiel manuell beendet.")
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Programm beendet.")

finally:
    env.close()