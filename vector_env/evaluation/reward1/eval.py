import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vector_env.agents.reward1.flappy_vector_env import FlappyBirdEnv


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

            obs, reward, done, info = env.step(action)
            
            step_counter += 1

            total_reward += reward

            score = info[0].get('score', 0)
            total_score = score

        scores_all_episodes.append(total_score)
        rewards_all_episodes.append(total_reward)
        
        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            print(f"Episode {ep + 1}/{n_episodes} abgeschlossen. Schritte: {step_counter}")

    return scores_all_episodes, rewards_all_episodes

def cumulative_average(data):
    """
    Gibt eine Liste zurück, in der der i-te Eintrag
    der Durchschnitt aller Datenpunkte bis einschließlich i ist.
    """
    cum_sum = np.cumsum(data)
    return cum_sum / np.arange(1, len(data) + 1)


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


if __name__ == "__main__":

    save_dir = os.path.join("vector_env", "evaluation", "reward1", "metriken1")
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

    model_dqn1 = DQN.load(r"vector_env\models\DQN\training6\best_model.zip")
    model_ppo1 = PPO.load(r"vector_env\models\PPO\training4\best_model.zip")

    num_episodes = 100  # Geändert von 1000 auf 100
    max_steps_per_episode = 1000  # Maximale Schritte pro Episode

    scores_dqn, rewards_dqn = evaluate_model(
        model_dqn1, env, num_episodes, max_steps_per_episode)
    scores_ppo, rewards_ppo = evaluate_model(
        model_ppo1, env, num_episodes, max_steps_per_episode)

    env.close()

    mean_scores_dqn = cumulative_average(scores_dqn)
    mean_rewards_dqn = cumulative_average(rewards_dqn)
    mean_scores_ppo = cumulative_average(scores_ppo)
    mean_rewards_ppo = cumulative_average(rewards_ppo)

    mean_dqn = np.mean(scores_dqn)
    std_dqn  = np.std(scores_dqn)
    mean_ppo = np.mean(scores_ppo)
    std_ppo  = np.std(scores_ppo)

    print(f"[SCORE] DQN: Mean = {mean_dqn:.2f}, Std = {std_dqn:.2f}")
    print(f"[SCORE] PPO: Mean = {mean_ppo:.2f}, Std = {std_ppo:.2f}")

    mean_reward_dqn = np.mean(rewards_dqn)
    std_reward_dqn  = np.std(rewards_dqn)
    mean_reward_ppo = np.mean(rewards_ppo)
    std_reward_ppo  = np.std(rewards_ppo)

    print(f"[REWARD] DQN: Mean = {mean_reward_dqn:.2f}, Std = {std_reward_dqn:.2f}")
    print(f"[REWARD] PPO: Mean = {mean_reward_ppo:.2f}, Std = {std_reward_ppo:.2f}")

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

    save_metrics_as_numpy(
        save_dir,
        scores_dqn, mean_scores_dqn, rewards_dqn, mean_rewards_dqn,
        scores_ppo, mean_scores_ppo, rewards_ppo, mean_rewards_ppo
    )

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

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=scores_dqn,
        name='DQN',
        boxmean='sd'
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

    fig_violin = px.violin(
        df_scores_rewards,
        y="Score",
        color="Algorithm",
        box=True,
        points="all",
        title="Violin Plot der Scores (DQN vs. PPO)"
    )
    fig_violin.update_layout(yaxis_title='Score')
    fig_violin.write_image(os.path.join(save_dir, "violin_scores.png"))
    fig_violin.show()


    fig_violin_reward = px.violin(
        df_scores_rewards,
        y="Reward",
        color="Algorithm",
        box=True,
        points="all", 
        title="Violin Plot der Rewards (DQN vs. PPO)"
    )
    fig_violin_reward.update_layout(yaxis_title='Reward')
    fig_violin_reward.write_image(os.path.join(save_dir, "violin_rewards.png"))
    fig_violin_reward.show()

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
        barmode='group'
    )
    fig_min_max_reward.write_image(os.path.join(save_dir, "min_max_rewards_comparison.png"))
    fig_min_max_reward.show()

    print("Done")
