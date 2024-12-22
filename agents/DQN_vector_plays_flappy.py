from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from envs.vector_env.flappy_vector_env import FlappyVectorEnv
from envs.vector_env.flappy_vector_env_render import FlappyVectorEnvWithRendering
import keyboard

# Umgebung einpacken
env = DummyVecEnv([lambda: FlappyVectorEnv])
# env = DummyVecEnv([lambda: FlappyVectorEnvWithRendering(render_mode="human")])

# Modell laden

model = DQN.load("dqn_flappy_bird_final.zip")

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
