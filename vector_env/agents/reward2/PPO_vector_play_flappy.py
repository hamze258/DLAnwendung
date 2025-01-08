from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from vector_env.agents.reward2.flappy_vector_env import FlappyBirdEnv
import keyboard

# Wrappen im VecEnv
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="human")])

# Gelerntes Modell laden
model = PPO.load(r"vector_env\models\PPO\training15\best_model.zip")

try:
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            #print(obs[0])
            if keyboard.is_pressed('q'):  # Prüfen, ob 'q' gedrückt wurde
                print("Spiel manuell beendet.")
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Programm beendet.")

finally:
    env.close()
