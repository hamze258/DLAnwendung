import time
import gym
from envs.vector_env.flappy_vector_env import FlappyVectorEnv
from envs.image_env.flappy_image_env import FlappyImageEnv

def play_environment(env_name="vector"):
    if env_name == "vector":
        env = FlappyVectorEnv()
    elif env_name == "image":
        env = FlappyImageEnv()
    else:
        raise ValueError("Invalid environment name. Use 'vector' or 'image'.")

    obs = env.reset()
    done = False
    print("Spiel gestartet! Benutze die Eingabe: 'w' für Flap, 'q' zum Beenden.")
    
    while not done:
        env.render()
        action = 0  # Standardmäßig keine Aktion

        # Benutzereingabe abfragen
        user_input = input("Aktion ('w' = Flap, 'q' = Beenden): ").lower()
        if user_input == 'w':
            action = 1  # Flap
        elif user_input == 'q':
            print("Spiel beendet!")
            break

        obs, reward, done, info = env.step(action)
        print(f"Beobachtung: {obs}, Belohnung: {reward}, Fertig: {done}")

        if done:
            print("Spiel vorbei!")
            obs = env.reset()
            done = False  # Starten Sie eine neue Episode

    env.close()

if __name__ == "__main__":
    # Wechsel zwischen Umgebungen möglich: "vector" oder "image"
    play_environment(env_name="vector")
