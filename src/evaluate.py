import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from envs.environment_image import FlappyBirdEnv
import matplotlib.pyplot as plt
import numpy as np

# Funktion zur Bewertung eines Modells
def evaluate_agent(model_path, env, num_episodes=100):
    model = PPO.load(model_path)  # Modell laden
    scores = []  # Liste zur Speicherung der Scores
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_score = 0  # Score der Episode

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # Tatsächlichen Score aus `info` extrahieren
            episode_score = info[0].get("score", episode_score)
        
        # Score dieser Episode speichern
        scores.append(episode_score)
    
    # Metriken berechnen
    average_score = sum(scores) / len(scores)
    high_score = max(scores)
    
    return average_score, high_score

# Umgebung erstellen
env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode=None)])

# Alle Modelle im Ordner "agents" finden
agents_dir = "src\\single_agent"
agent_files = [os.path.join(agents_dir, file) for file in os.listdir(agents_dir) if file.endswith(".zip")]

# Ergebnisse speichern
agent_names = []
highscores = []
mean_scores = []

# Agenten bewerten
for agent_file in agent_files:
    agent_name = os.path.basename(agent_file).split(".")[0]
    print(f"Bewerte Agenten: {agent_name}")
    
    print(agent_file)
    average_score, high_score = evaluate_agent(agent_file, env)
    
    agent_names.append(agent_name)
    mean_scores.append(average_score)
    highscores.append(high_score)
    
    print(f"{agent_name} -> Highscore: {high_score}, Mean Score: {average_score}")

# Ergebnisse plotten
plt.figure(figsize=(12, 6))
x = np.arange(len(agent_names))

bar_width = 0.4
plt.bar(x - bar_width / 2, highscores, width=bar_width, label="Highscore", color="green", alpha=0.7)
plt.bar(x + bar_width / 2, mean_scores, width=bar_width, label="Mean Score", color="blue", alpha=0.7)

plt.xlabel("Agents")
plt.ylabel("Scores")
plt.title("Comparison of Agents")
plt.xticks(x, agent_names, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# Umgebung schließen
env.close()
