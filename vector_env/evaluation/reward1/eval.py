import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import pandas as pd

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vector_env.agents.reward1.flappy_vector_env import FlappyBirdEnv

# ----------------------------------------------------
# Funktion zum Auswerten eines Modells
# ----------------------------------------------------
def evaluate_model(model, env, n_episodes=100, max_steps=10000):
    """
    Führt bis zu n_episodes Episoden im gegebenen (Vec-)Env durch,
    wobei jede Episode maximal max_steps Schritte ausführt.
    Gibt zwei Listen zurück:
      1. scores_all_episodes: Score pro Episode
      2. rewards_all_episodes: Gesamtreward pro Episode
    """
    scores_all_episodes = []
    rewards_all_episodes = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_score = 0
        total_reward = 0
        step_counter = 0

        while not done and step_counter < max_steps:
            action, _states = model.predict(obs, deterministic=True)

            # Schritt im Environment
            obs, reward, done, info = env.step(action)
            
            # Schrittzähler erhöhen
            step_counter += 1

            # Reward summieren
            total_reward += reward

            # Extrahiere den Score aus dem info-Dictionary
            # Annahme: 'score' ist im info-Dictionary enthalten
            score = info[0].get('score', 0)  # DummyVecEnv => info ist eine Liste
            total_score = score  # Update des Scores (falls kumulativ, ändern Sie dies entsprechend

        scores_all_episodes.append(total_score)
        rewards_all_episodes.append(total_reward)
        
        # Optional: Fortschritt anzeigen
        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            print(f"Episode {ep + 1}/{n_episodes} abgeschlossen. Schritte: {step_counter}")

    return scores_all_episodes, rewards_all_episodes

# ----------------------------------------------------
# Funktion für kumulative Durchschnittswerte
# ----------------------------------------------------
def cumulative_average(data):
    """
    Gibt eine Liste zurück, in der der i-te Eintrag
    der Durchschnitt aller Datenpunkte bis einschließlich i ist.
    """
    cum_sum = np.cumsum(data)
    return cum_sum / np.arange(1, len(data) + 1)

# ----------------------------------------------------
# Funktion zum Speichern der Metriken als NumPy-Arrays
# ----------------------------------------------------
def save_metrics_as_numpy(save_dir,
                          scores_dqn, mean_scores_dqn, rewards_dqn, mean_rewards_dqn,
                          scores_ppo, mean_scores_ppo, rewards_ppo, mean_rewards_ppo):
    """
    Speichert die Metriken als NumPy-Arrays in einer .npz-Datei.
    """
    np.savez(
        os.path.join(save_dir, "evaluation_metrics.npz"),
        scores_dqn=np.array(scores_dqn),
        mean_scores_dqn=np.array(mean_scores_dqn),
        rewards_dqn=np.array(rewards_dqn),
        mean_rewards_dqn=np.array(mean_rewards_dqn),
        scores_ppo=np.array(scores_ppo),
        mean_scores_ppo=np.array(mean_scores_ppo),
        rewards_ppo=np.array(rewards_ppo),
        mean_rewards_ppo=np.array(mean_rewards_ppo)
    )
    print(f"Metriken wurden als NumPy-Arrays in 'evaluation_metrics.npz' gespeichert.")

# ----------------------------------------------------
# Funktion zum Laden der gespeicherten NumPy-Arrays (optional)
# ----------------------------------------------------
def load_metrics_from_numpy(save_dir):
    """
    Lädt die Metriken aus einer .npz-Datei.
    """
    data = np.load(os.path.join(save_dir, "evaluation_metrics.npz"))
    scores_dqn = data['scores_dqn']
    mean_scores_dqn = data['mean_scores_dqn']
    rewards_dqn = data['rewards_dqn']
    mean_rewards_dqn = data['mean_rewards_dqn']
    scores_ppo = data['scores_ppo']
    mean_scores_ppo = data['mean_scores_ppo']
    rewards_ppo = data['rewards_ppo']
    mean_rewards_ppo = data['mean_rewards_ppo']
    return (scores_dqn, mean_scores_dqn, rewards_dqn, mean_rewards_dqn,
            scores_ppo, mean_scores_ppo, rewards_ppo, mean_rewards_ppo)

# ----------------------------------------------------
# Hauptteil: Vergleich von zwei Modellen
# ----------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # 0) Sicherstellen, dass das Verzeichnis existiert
    # ------------------------------------------------------------
    save_dir = os.path.join("vector_env", "evaluation", "reward1", "metriken")
    os.makedirs(save_dir, exist_ok=True)

    # 1) Environment erstellen (ggf. mit render=False)
    env = DummyVecEnv([lambda: FlappyBirdEnv()])

    # 2) Zwei vortrainierte Modelle laden
    model_dqn1 = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")
    model_ppo1 = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")

    # model_dqn2 = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")
    # model_ppo2 = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")

    # model_dqn3 = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")
    # model_ppo3 = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")

    # Falls Sie ein anderes zweites Modell vergleichen möchten, laden Sie es hier.

    # 3) Anzahl der Episoden für den Vergleich
    num_episodes = 100  # Geändert von 1000 auf 100
    max_steps_per_episode = 100000  # Maximale Schritte pro Episode

    # 4) Modelle auswerten
    scores_dqn, rewards_dqn = evaluate_model(
        model_dqn1, env, num_episodes, max_steps_per_episode)
    scores_ppo, rewards_ppo = evaluate_model(
        model_ppo1, env, num_episodes, max_steps_per_episode)

    # Environment schließen (nach Evaluation)
    env.close()

    # 5) Kumulative Durchschnittswerte berechnen
    mean_scores_dqn = cumulative_average(scores_dqn)
    mean_rewards_dqn = cumulative_average(rewards_dqn)
    mean_scores_ppo = cumulative_average(scores_ppo)
    mean_rewards_ppo = cumulative_average(rewards_ppo)

    # 6) Score-Statistiken berechnen
    mean_dqn = np.mean(scores_dqn)
    std_dqn  = np.std(scores_dqn)
    mean_ppo = np.mean(scores_ppo)
    std_ppo  = np.std(scores_ppo)

    print(f"[SCORE] DQN: Mean = {mean_dqn:.2f}, Std = {std_dqn:.2f}")
    print(f"[SCORE] PPO: Mean = {mean_ppo:.2f}, Std = {std_ppo:.2f}")

    # 7) Reward-Statistiken berechnen
    mean_reward_dqn = np.mean(rewards_dqn)
    std_reward_dqn  = np.std(rewards_dqn)
    mean_reward_ppo = np.mean(rewards_ppo)
    std_reward_ppo  = np.std(rewards_ppo)

    print(f"[REWARD] DQN: Mean = {mean_reward_dqn:.2f}, Std = {std_reward_dqn:.2f}")
    print(f"[REWARD] PPO: Mean = {mean_reward_ppo:.2f}, Std = {std_reward_ppo:.2f}")

    # 8) Zusätzliche Metriken zur Robustheit
    # Beispiel: Schlechteste und beste Episode
    min_score_dqn = np.min(scores_dqn)
    max_score_dqn = np.max(scores_dqn)
    min_score_ppo = np.min(scores_ppo)
    max_score_ppo = np.max(scores_ppo)

    min_reward_dqn = np.min(rewards_dqn)
    max_reward_dqn = np.max(rewards_dqn)
    min_reward_ppo = np.min(rewards_ppo)
    max_reward_ppo = np.max(rewards_ppo)

    print(f"DQN: Min Score = {min_score_dqn}, Max Score = {max_score_dqn}")
    print(f"PPO: Min Score = {min_score_ppo}, Max Score = {max_score_ppo}")
    print(f"DQN: Min Reward = {min_reward_dqn}, Max Reward = {max_reward_dqn}")
    print(f"PPO: Min Reward = {min_reward_ppo}, Max Reward = {max_reward_ppo}")

    # ------------------------------------------------------------
    # Speichern der Metriken als NumPy-Arrays
    # ------------------------------------------------------------
    save_metrics_as_numpy(
        save_dir,
        scores_dqn, mean_scores_dqn, rewards_dqn, mean_rewards_dqn,
        scores_ppo, mean_scores_ppo, rewards_ppo, mean_rewards_ppo
    )

    # ------------------------------------------------------------
    # DataFrame für Plotly (Scores und Rewards pro Episode, Algorithmus)
    # ------------------------------------------------------------
    df_dqn = pd.DataFrame({
        "Episode": range(1, len(scores_dqn) + 1),
        "Score": scores_dqn,
        "Mean Score": mean_scores_dqn,
        "Reward": rewards_dqn,
        "Mean Reward": mean_rewards_dqn,
        "Algorithm": ["DQN"] * len(scores_dqn)
    })

    df_ppo = pd.DataFrame({
        "Episode": range(1, len(scores_ppo) + 1),
        "Score": scores_ppo,
        "Mean Score": mean_scores_ppo,
        "Reward": rewards_ppo,
        "Mean Reward": mean_rewards_ppo,
        "Algorithm": ["PPO"] * len(scores_ppo)
    })

    df_scores_rewards = pd.concat([df_dqn, df_ppo], ignore_index=True)

    # ------------------------------------------------------------
    # Plot 1: Boxplot für beide Algorithmen (Score-Verteilung)
    # ------------------------------------------------------------
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=scores_dqn,
        name='DQN',
        boxmean='sd'   # zeigt Durchschnitt + SD
    ))
    fig_box.add_trace(go.Box(
        y=scores_ppo,
        name='PPO',
        boxmean='sd'
    ))
    fig_box.update_layout(
        title='Boxplot: DQN vs. PPO (Scores)',
        yaxis_title='Score'
    )
    fig_box.write_image(os.path.join(save_dir, "boxplot_scores.png"))
    fig_box.show()

    # ------------------------------------------------------------
    # Plot 2: Boxplot für beide Algorithmen (Reward-Verteilung)
    # ------------------------------------------------------------
    fig_box_reward = go.Figure()
    fig_box_reward.add_trace(go.Box(
        y=rewards_dqn,
        name='DQN',
        boxmean='sd'
    ))
    fig_box_reward.add_trace(go.Box(
        y=rewards_ppo,
        name='PPO',
        boxmean='sd'
    ))
    fig_box_reward.update_layout(
        title='Boxplot: DQN vs. PPO (Rewards)',
        yaxis_title='Reward'
    )
    fig_box_reward.write_image(os.path.join(save_dir, "boxplot_rewards.png"))
    fig_box_reward.show()

    # ------------------------------------------------------------
    # Plot 3: ECDF (Empirical Cumulative Distribution Function) für Scores
    # ------------------------------------------------------------
    fig_ecdf = px.ecdf(
        df_scores_rewards,
        x="Score",
        color="Algorithm",
        title="ECDF: DQN vs. PPO (Scores)"
    )
    fig_ecdf.update_layout(
        xaxis_title='Score',
        yaxis_title='Kumulative Wahrscheinlichkeit'
    )
    fig_ecdf.write_image(os.path.join(save_dir, "ecdf_scores.png"))
    fig_ecdf.show()

    # ------------------------------------------------------------
    # Plot 4: ECDF (Empirical Cumulative Distribution Function) für Rewards
    # ------------------------------------------------------------
    fig_ecdf_reward = px.ecdf(
        df_scores_rewards,
        x="Reward",
        color="Algorithm",
        title="ECDF: DQN vs. PPO (Rewards)"
    )
    fig_ecdf_reward.update_layout(
        xaxis_title='Reward',
        yaxis_title='Kumulative Wahrscheinlichkeit'
    )
    fig_ecdf_reward.write_image(os.path.join(save_dir, "ecdf_rewards.png"))
    fig_ecdf_reward.show()

    # ------------------------------------------------------------
    # Plot 5: Violin-Plot (Scores)
    # ------------------------------------------------------------
    fig_violin = px.violin(
        df_scores_rewards,
        y="Score",
        color="Algorithm",
        box=True,        # zeigt zusätzlich den Boxplot
        points="all",    # alle Datenpunkte
        title="Violin Plot der Scores (DQN vs. PPO)"
    )
    fig_violin.update_layout(yaxis_title='Score')
    fig_violin.write_image(os.path.join(save_dir, "violin_scores.png"))
    fig_violin.show()

    # ------------------------------------------------------------
    # Plot 6: Violin-Plot (Rewards)
    # ------------------------------------------------------------
    fig_violin_reward = px.violin(
        df_scores_rewards,
        y="Reward",
        color="Algorithm",
        box=True,        # zeigt zusätzlich den Boxplot
        points="all",    # alle Datenpunkte
        title="Violin Plot der Rewards (DQN vs. PPO)"
    )
    fig_violin_reward.update_layout(yaxis_title='Reward')
    fig_violin_reward.write_image(os.path.join(save_dir, "violin_rewards.png"))
    fig_violin_reward.show()

    # ------------------------------------------------------------
    # Plot 7: Liniendiagramm: Score pro Episode (ungeglättet)
    # ------------------------------------------------------------
    fig_line_episodes = px.line(
        df_scores_rewards,
        x="Episode",
        y="Score",
        color="Algorithm",
        title="Scores pro Episode (DQN vs. PPO)"
    )
    fig_line_episodes.update_layout(xaxis_title='Episode', yaxis_title='Score')
    fig_line_episodes.write_image(os.path.join(save_dir, "line_episode_scores.png"))
    fig_line_episodes.show()

    # ------------------------------------------------------------
    # Plot 8: Liniendiagramm: Reward pro Episode (ungeglättet)
    # ------------------------------------------------------------
    fig_line_rewards = px.line(
        df_scores_rewards,
        x="Episode",
        y="Reward",
        color="Algorithm",
        title="Rewards pro Episode (DQN vs. PPO)"
    )
    fig_line_rewards.update_layout(xaxis_title='Episode', yaxis_title='Reward')
    fig_line_rewards.write_image(os.path.join(save_dir, "line_episode_rewards.png"))
    fig_line_rewards.show()

    # ------------------------------------------------------------
    # Plot 9: Liniendiagramm kumulativer Durchschnitt (Scores)
    # ------------------------------------------------------------
    fig_line_cum = go.Figure()
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(mean_scores_dqn) + 1)),
        y=mean_scores_dqn,
        mode='lines',
        name='DQN (Kumul. Avg)'
    ))
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(mean_scores_ppo) + 1)),
        y=mean_scores_ppo,
        mode='lines',
        name='PPO (Kumul. Avg)'
    ))
    fig_line_cum.update_layout(
        title='Kumulative Durchschnitts-Scores pro Episode (DQN vs. PPO)',
        xaxis_title='Episode',
        yaxis_title='Kumulativer Durchschnitts-Score'
    )
    fig_line_cum.write_image(os.path.join(save_dir, "line_cumulative_avg_scores.png"))
    fig_line_cum.show()

    # ------------------------------------------------------------
    # Plot 10: Liniendiagramm kumulativer Durchschnitt (Rewards)
    # ------------------------------------------------------------
    fig_line_cum_reward = go.Figure()
    fig_line_cum_reward.add_trace(go.Scatter(
        x=list(range(1, len(mean_rewards_dqn) + 1)),
        y=mean_rewards_dqn,
        mode='lines',
        name='DQN (Kumul. Avg)'
    ))
    fig_line_cum_reward.add_trace(go.Scatter(
        x=list(range(1, len(mean_rewards_ppo) + 1)),
        y=mean_rewards_ppo,
        mode='lines',
        name='PPO (Kumul. Avg)'
    ))
    fig_line_cum_reward.update_layout(
        title='Kumulative Durchschnitts-Rewards pro Episode (DQN vs. PPO)',
        xaxis_title='Episode',
        yaxis_title='Kumulativer Durchschnitts-Reward'
    )
    fig_line_cum_reward.write_image(os.path.join(save_dir, "line_cumulative_avg_rewards.png"))
    fig_line_cum_reward.show()

    # ------------------------------------------------------------
    # Plot 11: Vergleich der minimalen und maximalen Scores
    # ------------------------------------------------------------
    categories = ["Min Score", "Max Score"]
    dqn_values = [min_score_dqn, max_score_dqn]
    ppo_values = [min_score_ppo, max_score_ppo]

    fig_min_max = go.Figure(data=[
        go.Bar(name='DQN', x=categories, y=dqn_values, marker_color='skyblue'),
        go.Bar(name='PPO', x=categories, y=ppo_values, marker_color='salmon')
    ])
    fig_min_max.update_layout(
        title='Vergleich der minimalen und maximalen Scores',
        xaxis_title='Kategorie',
        yaxis_title='Score',
        barmode='group'  # Nebeneinander statt gestapelt
    )
    fig_min_max.write_image(os.path.join(save_dir, "min_max_scores_comparison.png"))
    fig_min_max.show()

    # ------------------------------------------------------------
    # Plot 12: Vergleich der minimalen und maximalen Rewards
    # ------------------------------------------------------------
    categories_reward = ["Min Reward", "Max Reward"]
    dqn_values_reward = [min_reward_dqn, max_reward_dqn]
    ppo_values_reward = [min_reward_ppo, max_reward_ppo]

    fig_min_max_reward = go.Figure(data=[
        go.Bar(name='DQN', x=categories_reward, y=dqn_values_reward, marker_color='lightgreen'),
        go.Bar(name='PPO', x=categories_reward, y=ppo_values_reward, marker_color='orange')
    ])
    fig_min_max_reward.update_layout(
        title='Vergleich der minimalen und maximalen Rewards',
        xaxis_title='Kategorie',
        yaxis_title='Reward',
        barmode='group'  # Nebeneinander statt gestapelt
    )
    fig_min_max_reward.write_image(os.path.join(save_dir, "min_max_rewards_comparison.png"))
    fig_min_max_reward.show()

    # Optional: Weitere Plots oder Metriken hinzufügen

    print("Alle Vergleichsmetriken wurden geplottet und im Verzeichnis 'diagrams' gespeichert.")
