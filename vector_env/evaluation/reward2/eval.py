import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os  # Für das Anlegen von Verzeichnissen
import pandas as pd  # Für komfortable DataFrame-Erzeugung (optional)

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vector_env.agents.flappy_vector_env import FlappyBirdEnv

# ----------------------------------------------------
# Funktion zum Auswerten eines Modells
# ----------------------------------------------------
def evaluate_model(model, env, n_episodes=1000):
    """
    Führt n_episodes Episoden im gegebenen (Vec-)Env durch
    und gibt zwei Listen zurück:
      1. rewards_all_episodes: Summe der Rewards pro Episode
      2. inference_times: Zeit (in Sekunden) für jeden predict()-Aufruf
    """
    rewards_all_episodes = []
    inference_times = []  # Speichert die Zeit pro predict()-Aufruf
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            start_time = time.perf_counter()
            action, _states = model.predict(obs, deterministic=True)
            end_time = time.perf_counter()
            
            # Inference-Zeit für diesen Schritt
            inference_times.append(end_time - start_time)
            
            # Schritt im Environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # DummyVecEnv => reward ist Array
        
        rewards_all_episodes.append(total_reward)
    
    return rewards_all_episodes, inference_times


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
# Hauptteil: Vergleich von DQN und PPO
# ----------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # 0) Sicherstellen, dass das Verzeichnis existiert
    # ------------------------------------------------------------
    save_dir = r"vector_env\evaluation\train1\diagrams"
    os.makedirs(save_dir, exist_ok=True)

    # 1) Environment erstellen (ggf. mit render=False)
    env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

    # 2) Zwei vortrainierte Modelle laden
    model_dqn = DQN.load(r"vector_env\models\DQN\training4\best_model.zip")
    model_ppo = PPO.load(r"vector_env\models\PPO\training1\best_model.zip")

    # 3) Anzahl der Episoden für den Vergleich
    num_episodes = 100

    # 4) Modelle auswerten
    rewards_dqn, inference_times_dqn = evaluate_model(model_dqn, env, num_episodes)
    rewards_ppo, inference_times_ppo = evaluate_model(model_ppo, env, num_episodes)

    # Environment schließen (nach Evaluation)
    env.close()

    # 5) Belohnungs-Statistiken berechnen
    mean_dqn = np.mean(rewards_dqn)
    std_dqn  = np.std(rewards_dqn)
    mean_ppo = np.mean(rewards_ppo)
    std_ppo  = np.std(rewards_ppo)

    print(f"[REWARD] DQN: Mean = {mean_dqn:.2f}, Std = {std_dqn:.2f}")
    print(f"[REWARD] PPO: Mean = {mean_ppo:.2f}, Std = {std_ppo:.2f}")

    # 6) Inference-Time-Statistiken
    mean_inf_dqn = np.mean(inference_times_dqn)
    std_inf_dqn  = np.std(inference_times_dqn)
    mean_inf_ppo = np.mean(inference_times_ppo)
    std_inf_ppo  = np.std(inference_times_ppo)

    print(f"[INFERENCE TIME] DQN: Mean = {mean_inf_dqn*1e3:.4f} ms, Std = {std_inf_dqn*1e3:.4f} ms")
    print(f"[INFERENCE TIME] PPO: Mean = {mean_inf_ppo*1e3:.4f} ms, Std = {std_inf_ppo*1e3:.4f} ms")

    # ------------------------------------------------------------
    # DataFrame für Plotly (Rewards pro Episode, Algorithm)
    # ------------------------------------------------------------
    df_dqn = pd.DataFrame({
        "Episode": range(1, len(rewards_dqn) + 1),
        "Reward": rewards_dqn,
        "Algorithm": ["DQN"] * len(rewards_dqn)
    })

    df_ppo = pd.DataFrame({
        "Episode": range(1, len(rewards_ppo) + 1),
        "Reward": rewards_ppo,
        "Algorithm": ["PPO"] * len(rewards_ppo)
    })

    df_rewards = pd.concat([df_dqn, df_ppo], ignore_index=True)

    # ------------------------------------------------------------
    #  Plot 1: Boxplot für beide Algorithmen (Reward-Verteilung)
    # ------------------------------------------------------------
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=rewards_dqn,
        name='DQN',
        boxmean='sd'   # zeigt Durchschnitt + SD
    ))
    fig_box.add_trace(go.Box(
        y=rewards_ppo,
        name='PPO',
        boxmean='sd'
    ))
    fig_box.update_layout(
        title='Boxplot: DQN vs. PPO (Rewards)',
        yaxis_title='Rewards'
    )
    fig_box.write_image(f"{save_dir}/boxplot_rewards.png")
    fig_box.show()

    # ------------------------------------------------------------
    # Plot 2: Gemeinsames Histogramm (Rewards)
    # ------------------------------------------------------------
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=rewards_dqn, 
        name='DQN', 
        opacity=0.7
    ))
    fig_hist.add_trace(go.Histogram(
        x=rewards_ppo, 
        name='PPO', 
        opacity=0.7
    ))
    fig_hist.update_layout(
        barmode='overlay',
        title='Histogramm: DQN vs. PPO (Rewards)',
        xaxis_title='Rewards',
        yaxis_title='Anzahl'
    )
    fig_hist.write_image(f"{save_dir}/histogram_rewards.png")
    fig_hist.show()

    # ------------------------------------------------------------
    # Plot 3: ECDF (Empirical Cumulative Distribution Function)
    # ------------------------------------------------------------
    fig_ecdf = px.ecdf(
        df_rewards,
        x="Reward",
        color="Algorithm",
        title="ECDF: DQN vs. PPO (Rewards)"
    )
    fig_ecdf.update_layout(
        xaxis_title='Reward',
        yaxis_title='Kumulative Wahrscheinlichkeit'
    )
    fig_ecdf.write_image(f"{save_dir}/ecdf_rewards.png")
    fig_ecdf.show()

    # ------------------------------------------------------------
    # Plot 4: Violin-Plot (Rewards)
    # ------------------------------------------------------------
    fig_violin = px.violin(
        df_rewards,
        y="Reward",
        color="Algorithm",
        box=True,        # zeigt zusätzlich den Boxplot
        points="all",    # alle Datenpunkte
        title="Violin Plot der Rewards (DQN vs. PPO)"
    )
    fig_violin.update_layout(yaxis_title='Rewards')
    fig_violin.write_image(f"{save_dir}/violin_rewards.png")
    fig_violin.show()

    # ------------------------------------------------------------
    # Plot 5a: Liniendiagramm: Reward pro Episode (ungeglättet)
    # ------------------------------------------------------------
    # Damit sieht man, wie sich der Reward bei jeder Episode entwickelt hat.
    fig_line_episodes = px.line(
        df_rewards,
        x="Episode",
        y="Reward",
        color="Algorithm",
        title="Rewards pro Episode (DQN vs. PPO)"
    )
    fig_line_episodes.update_layout(xaxis_title='Episode', yaxis_title='Reward')
    fig_line_episodes.write_image(f"{save_dir}/line_episode_rewards.png")
    fig_line_episodes.show()

    # ------------------------------------------------------------
    # Plot 5b: Liniendiagramm kumulativer Durchschnitt (Rewards)
    # (falls weiterhin gewünscht)
    # ------------------------------------------------------------
    cum_avg_dqn = cumulative_average(rewards_dqn)
    cum_avg_ppo = cumulative_average(rewards_ppo)

    fig_line_cum = go.Figure()
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(cum_avg_dqn) + 1)),
        y=cum_avg_dqn,
        mode='lines',
        name='DQN (Cumulative Avg)'
    ))
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(cum_avg_ppo) + 1)),
        y=cum_avg_ppo,
        mode='lines',
        name='PPO (Cumulative Avg)'
    ))
    fig_line_cum.update_layout(
        title='Kumulative Durchschnitts-Rewards pro Episode (DQN vs. PPO)',
        xaxis_title='Episode',
        yaxis_title='Kumulativer Durchschnitts-Reward'
    )
    fig_line_cum.write_image(f"{save_dir}/line_cumulative_avg.png")
    fig_line_cum.show()

    # ------------------------------------------------------------
    # Neu: Boxplot der Inference-Zeiten
    # ------------------------------------------------------------
    times_dqn_ms = [t * 1e3 for t in inference_times_dqn]
    times_ppo_ms = [t * 1e3 for t in inference_times_ppo]

    fig_inf_box = go.Figure()
    fig_inf_box.add_trace(go.Box(
        y=times_dqn_ms,
        name='DQN',
        boxmean='sd'
    ))
    fig_inf_box.add_trace(go.Box(
        y=times_ppo_ms,
        name='PPO',
        boxmean='sd'
    ))
    fig_inf_box.update_layout(
        title='Inference Time (ms): DQN vs. PPO',
        yaxis_title='Zeit (ms)'
    )
    fig_inf_box.write_image(f"{save_dir}/boxplot_inference_times.png")
    fig_inf_box.show()

    # ------------------------------------------------------------
    # Optional: Histogramm der Inference-Zeiten
    # ------------------------------------------------------------
    fig_inf_hist = go.Figure()
    fig_inf_hist.add_trace(go.Histogram(
        x=times_dqn_ms,
        name='DQN',
        opacity=0.7
    ))
    fig_inf_hist.add_trace(go.Histogram(
        x=times_ppo_ms,
        name='PPO',
        opacity=0.7
    ))
    fig_inf_hist.update_layout(
        barmode='overlay',
        title='Histogramm: Inference Time (ms)',
        xaxis_title='Zeit (ms)',
        yaxis_title='Anzahl'
    )
    fig_inf_hist.write_image(f"{save_dir}/histogram_inference_times.png")
    fig_inf_hist.show()
