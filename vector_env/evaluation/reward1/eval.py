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
def evaluate_model(model, env, n_episodes=1000):
    """
    Führt n_episodes Episoden im gegebenen (Vec-)Env durch
    und gibt zwei Listen zurück:
      1. scores_all_episodes: Score pro Episode
      2. inference_times: Zeit (in Mikrosekunden) für jeden predict()-Aufruf
    """
    scores_all_episodes = []
    inference_times = []  # Speichert die Zeit pro predict()-Aufruf in Mikrosekunden

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_score = 0

        while not done:
            start_time = time.perf_counter()
            action, _states = model.predict(obs, deterministic=True)
            end_time = time.perf_counter()

            # Inference-Zeit für diesen Schritt in Mikrosekunden
            inference_time_us = (end_time - start_time) * 1e6
            inference_times.append(inference_time_us)

            # Schritt im Environment
            obs, reward, done, info = env.step(action)
            
            # Extrahiere den Score aus dem info-Dictionary
            # Annahme: 'score' ist im info-Dictionary enthalten
            score = info[0].get('score', 0)  # DummyVecEnv => info ist eine Liste
            total_score = score  # Update des Scores (falls kumulativ, ändern Sie dies entsprechend

        scores_all_episodes.append(total_score)
        
    return scores_all_episodes, inference_times  #, success_rate

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
def save_metrics_as_numpy(save_dir, scores_dqn, scores_ppo, inference_times_dqn_us, inference_times_ppo_us):
    """
    Speichert die Metriken als NumPy-Arrays in einer .npz-Datei.
    """
    np.savez(
        os.path.join(save_dir, "evaluation_metrics.npz"),
        scores_dqn=np.array(scores_dqn),
        scores_ppo=np.array(scores_ppo),
        inference_times_dqn_us=np.array(inference_times_dqn_us),
        inference_times_ppo_us=np.array(inference_times_ppo_us)
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
    scores_ppo = data['scores_ppo']
    inference_times_dqn_us = data['inference_times_dqn_us']
    inference_times_ppo_us = data['inference_times_ppo_us']
    return scores_dqn, scores_ppo, inference_times_dqn_us, inference_times_ppo_us

# ----------------------------------------------------
# Hauptteil: Vergleich von zwei Modellen
# ----------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    # 0) Sicherstellen, dass das Verzeichnis existiert
    # ------------------------------------------------------------
    save_dir = os.path.join("vector_env", "evaluation", "reward1", "diagrams")
    os.makedirs(save_dir, exist_ok=True)

    # 1) Environment erstellen (ggf. mit render=False)
    env = DummyVecEnv([lambda: FlappyBirdEnv()])

    # 2) Zwei vortrainierte Modelle laden
    model_dqn = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")
    model_ppo = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")
    # Falls Sie ein anderes zweites Modell vergleichen möchten, laden Sie es hier.

    # 3) Anzahl der Episoden für den Vergleich
    num_episodes = 1000  # Erhöht für robustere Statistiken

    # 4) Modelle auswerten
    scores_dqn, inference_times_dqn = evaluate_model(model_dqn, env, num_episodes)
    scores_ppo, inference_times_ppo = evaluate_model(model_ppo, env, num_episodes)

    # Environment schließen (nach Evaluation)
    env.close()

    # Umwandlung der Listen in NumPy-Arrays (falls noch nicht geschehen)
    scores_dqn = np.array(scores_dqn)
    scores_ppo = np.array(scores_ppo)
    inference_times_dqn_us = np.array(inference_times_dqn)
    inference_times_ppo_us = np.array(inference_times_ppo)

    # 5) Score-Statistiken berechnen
    mean_dqn = np.mean(scores_dqn)
    std_dqn  = np.std(scores_dqn)
    mean_ppo = np.mean(scores_ppo)
    std_ppo  = np.std(scores_ppo)

    print(f"[SCORE] DQN: Mean = {mean_dqn:.2f}, Std = {std_dqn:.2f}")
    print(f"[SCORE] PPO: Mean = {mean_ppo:.2f}, Std = {std_ppo:.2f}")

    # 6) Inference-Time-Statistiken
    mean_inf_dqn = np.mean(inference_times_dqn_us)
    std_inf_dqn  = np.std(inference_times_dqn_us)
    mean_inf_ppo = np.mean(inference_times_ppo_us)
    std_inf_ppo  = np.std(inference_times_ppo_us)

    print(f"[INFERENCE TIME] DQN: Mean = {mean_inf_dqn:.2f} µs, Std = {std_inf_dqn:.2f} µs")
    print(f"[INFERENCE TIME] PPO: Mean = {mean_inf_ppo:.2f} µs, Std = {std_inf_ppo:.2f} µs")

    # Zusätzliche Metriken zur Robustheit
    # Beispiel: Schlechteste und beste Episode
    min_score_dqn = np.min(scores_dqn)
    max_score_dqn = np.max(scores_dqn)
    min_score_ppo = np.min(scores_ppo)
    max_score_ppo = np.max(scores_ppo)

    print(f"DQN: Min Score = {min_score_dqn}, Max Score = {max_score_dqn}")
    print(f"PPO: Min Score = {min_score_ppo}, Max Score = {max_score_ppo}")

    # ------------------------------------------------------------
    # Speichern der Metriken als NumPy-Arrays
    # ------------------------------------------------------------
    save_metrics_as_numpy(
        save_dir,
        scores_dqn,
        scores_ppo,
        inference_times_dqn_us,
        inference_times_ppo_us
    )

    # ------------------------------------------------------------
    # DataFrame für Plotly (Scores pro Episode, Algorithmus)
    # ------------------------------------------------------------
    df_dqn = pd.DataFrame({
        "Episode": range(1, len(scores_dqn) + 1),
        "Score": scores_dqn,
        "Algorithm": ["DQN"] * len(scores_dqn)
    })

    df_ppo = pd.DataFrame({
        "Episode": range(1, len(scores_ppo) + 1),
        "Score": scores_ppo,
        "Algorithm": ["PPO"] * len(scores_ppo)
    })

    df_scores = pd.concat([df_dqn, df_ppo], ignore_index=True)

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
    # Plot 2: Gemeinsames Histogramm (Scores)
    # ------------------------------------------------------------
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=scores_dqn, 
        name='DQN', 
        opacity=0.7
    ))
    fig_hist.add_trace(go.Histogram(
        x=scores_ppo, 
        name='PPO', 
        opacity=0.7
    ))
    fig_hist.update_layout(
        barmode='overlay',
        title='Histogramm: DQN vs. PPO (Scores)',
        xaxis_title='Score',
        yaxis_title='Anzahl',
        bargap=0.2
    )
    fig_hist.update_traces(opacity=0.75)
    fig_hist.write_image(os.path.join(save_dir, "histogram_scores.png"))
    fig_hist.show()

    # ------------------------------------------------------------
    # Plot 3: ECDF (Empirical Cumulative Distribution Function)
    # ------------------------------------------------------------
    fig_ecdf = px.ecdf(
        df_scores,
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
    # Plot 4: Violin-Plot (Scores)
    # ------------------------------------------------------------
    fig_violin = px.violin(
        df_scores,
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
    # Plot 5a: Liniendiagramm: Score pro Episode (ungeglättet)
    # ------------------------------------------------------------
    # Damit sieht man, wie sich der Score bei jeder Episode entwickelt hat.
    fig_line_episodes = px.line(
        df_scores,
        x="Episode",
        y="Score",
        color="Algorithm",
        title="Scores pro Episode (DQN vs. PPO)"
    )
    fig_line_episodes.update_layout(xaxis_title='Episode', yaxis_title='Score')
    fig_line_episodes.write_image(os.path.join(save_dir, "line_episode_scores.png"))
    fig_line_episodes.show()

    # ------------------------------------------------------------
    # Plot 5b: Liniendiagramm kumulativer Durchschnitt (Scores)
    # ------------------------------------------------------------
    cum_avg_dqn = cumulative_average(scores_dqn)
    cum_avg_ppo = cumulative_average(scores_ppo)

    fig_line_cum = go.Figure()
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(cum_avg_dqn) + 1)),
        y=cum_avg_dqn,
        mode='lines',
        name='DQN (Kumul. Avg)'
    ))
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(cum_avg_ppo) + 1)),
        y=cum_avg_ppo,
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
    # Plot 6: Boxplot der Inference-Zeiten (Mikrosekunden)
    # ------------------------------------------------------------
    fig_inf_box = go.Figure()
    fig_inf_box.add_trace(go.Box(
        y=inference_times_dqn_us,
        name='DQN',
        boxmean='sd'
    ))
    fig_inf_box.add_trace(go.Box(
        y=inference_times_ppo_us,
        name='PPO',
        boxmean='sd'
    ))
    fig_inf_box.update_layout(
        title='Inference Time (µs): DQN vs. PPO',
        yaxis_title='Zeit (µs)'  # Aktualisierte Achsenbeschriftung
    )
    fig_inf_box.write_image(os.path.join(save_dir, "boxplot_inference_times_us.png"))
    fig_inf_box.show()

    # ------------------------------------------------------------
    # Plot 7: Histogramm der Inference-Zeiten (Mikrosekunden)
    # ------------------------------------------------------------
    fig_inf_hist = go.Figure()
    fig_inf_hist.add_trace(go.Histogram(
        x=inference_times_dqn_us,
        name='DQN',
        opacity=0.7
    ))
    fig_inf_hist.add_trace(go.Histogram(
        x=inference_times_ppo_us,
        name='PPO',
        opacity=0.7
    ))
    fig_inf_hist.update_layout(
        barmode='overlay',
        title='Histogramm: Inference Time (µs)',
        xaxis_title='Zeit (µs)',
        yaxis_title='Anzahl',
        bargap=0.2
    )
    fig_inf_hist.update_traces(opacity=0.75)
    fig_inf_hist.write_image(os.path.join(save_dir, "histogram_inference_times_us.png"))
    fig_inf_hist.show()

    # ------------------------------------------------------------
    # Zusätzliche Plots für Robustheitsmetriken
    # ------------------------------------------------------------

    # Plot 8: Vergleich der minimalen und maximalen Scores
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
        barmode='group'
    )
    fig_min_max.write_image(os.path.join(save_dir, "min_max_scores_comparison.png"))
    fig_min_max.show()

    # Optional: Weitere Robustheitsmetriken hinzufügen

    print("Alle Vergleichsmetriken wurden geplottet und im Verzeichnis 'diagrams1' gespeichert.")
