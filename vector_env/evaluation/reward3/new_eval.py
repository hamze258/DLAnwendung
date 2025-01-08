import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vector_env.agents.reward3.flappy_vector_env import FlappyBirdEnv

def evaluate_model(model, env, n_episodes=100, max_steps=10000):
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
    cum_sum = np.cumsum(data)
    return cum_sum / np.arange(1, len(data) + 1)

def save_metrics_as_numpy(save_dir, algorithm, scores, mean_scores, rewards, mean_rewards):
    np.savez(
        os.path.join(save_dir, f"evaluation_metrics_{algorithm}.npz"),
        scores=np.array(scores),
        mean_scores=np.array(mean_scores),
        rewards=np.array(rewards),
        mean_rewards=np.array(mean_rewards)
    )
    print(f"Metriken für {algorithm} wurden als NumPy-Arrays in 'evaluation_metrics_{algorithm}.npz' gespeichert.")

def load_metrics_from_numpy(save_dir, algorithm):
    data = np.load(os.path.join(save_dir, f"evaluation_metrics_{algorithm}.npz"))
    scores = data['scores']
    mean_scores = data['mean_scores']
    rewards = data['rewards']
    mean_rewards = data['mean_rewards']
    return scores, mean_scores, rewards, mean_rewards

if __name__ == "__main__":

    selected_algorithm = 'PPO'

    # Validierung der Auswahl
    if selected_algorithm not in ['DQN', 'PPO']:
        raise ValueError("selected_algorithm muss entweder 'DQN' oder 'PPO' sein.")

    save_dir = os.path.join("vector_env", "evaluation", "reward3", "PPO")
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

    if selected_algorithm == 'DQN':
        model = DQN.load(r"vector_env\models\DQN\training8\best_model.zip")
    else:
        model = PPO.load(r"vector_env\models\PPO\training16\best_model.zip")

    num_episodes = 100
    max_steps_per_episode = 1000

    scores, rewards = evaluate_model(model, env, num_episodes, max_steps_per_episode)

    # Environment schließen (nach Evaluation)
    env.close()

    mean_scores = cumulative_average(scores)
    mean_rewards = cumulative_average(rewards)

    mean_score = np.mean(scores)
    std_score  = np.std(scores)

    print(f"[SCORE] {selected_algorithm}: Mean = {mean_score:.2f}, Std = {std_score:.2f}")

    mean_reward = np.mean(rewards)
    std_reward  = np.std(rewards)

    print(f"[REWARD] {selected_algorithm}: Mean = {mean_reward:.2f}, Std = {std_reward:.2f}")

    min_score = np.min(scores)
    max_score = np.max(scores)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"{selected_algorithm}: Min Score = {min_score}, Max Score = {max_score}")
    print(f"{selected_algorithm}: Min Reward = {min_reward}, Max Reward = {max_reward}")

    save_metrics_as_numpy(
        save_dir,
        selected_algorithm,
        scores, mean_scores, rewards, mean_rewards
    )

    df = pd.DataFrame({
        "Episode": range(1, len(scores) + 1),
        "Score": scores,
        "Mean Score": mean_scores,
        "Reward": rewards,
        "Mean Reward": mean_rewards
    })

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=df["Score"],
        name=selected_algorithm,
        boxmean='sd'
    ))
    fig_box.update_layout(
        title=f'Boxplot: {selected_algorithm} (Scores)',
        yaxis_title='Score'
    )
    fig_box.write_image(os.path.join(save_dir, f"boxplot_scores_{selected_algorithm}.png"))
    fig_box.show()

    fig_box_reward = go.Figure()
    fig_box_reward.add_trace(go.Box(
        y=df["Reward"],
        name=selected_algorithm,
        boxmean='sd'
    ))
    fig_box_reward.update_layout(
        title=f'Boxplot: {selected_algorithm} (Rewards)',
        yaxis_title='Reward'
    )
    fig_box_reward.write_image(os.path.join(save_dir, f"boxplot_rewards_{selected_algorithm}.png"))
    fig_box_reward.show()

    fig_ecdf = px.ecdf(
        df,
        x="Score",
        title=f"ECDF: {selected_algorithm} (Scores)"
    )
    fig_ecdf.update_layout(
        xaxis_title='Score',
        yaxis_title='Kumulative Wahrscheinlichkeit'
    )
    fig_ecdf.write_image(os.path.join(save_dir, f"ecdf_scores_{selected_algorithm}.png"))
    fig_ecdf.show()

    fig_ecdf_reward = px.ecdf(
        df,
        x="Reward",
        title=f"ECDF: {selected_algorithm} (Rewards)"
    )
    fig_ecdf_reward.update_layout(
        xaxis_title='Reward',
        yaxis_title='Kumulative Wahrscheinlichkeit'
    )
    fig_ecdf_reward.write_image(os.path.join(save_dir, f"ecdf_rewards_{selected_algorithm}.png"))
    fig_ecdf_reward.show()

    fig_violin = px.violin(
        df,
        y="Score",
        box=True,
        points="all",
        title=f"Violin Plot der Scores ({selected_algorithm})"
    )
    fig_violin.update_layout(yaxis_title='Score')
    fig_violin.write_image(os.path.join(save_dir, f"violin_scores_{selected_algorithm}.png"))
    fig_violin.show()

    fig_violin_reward = px.violin(
        df,
        y="Reward",
        box=True,
        points="all",
        title=f"Violin Plot der Rewards ({selected_algorithm})"
    )
    fig_violin_reward.update_layout(yaxis_title='Reward')
    fig_violin_reward.write_image(os.path.join(save_dir, f"violin_rewards_{selected_algorithm}.png"))
    fig_violin_reward.show()

    fig_line_episodes = px.line(
        df,
        x="Episode",
        y="Score",
        title=f"Scores pro Episode ({selected_algorithm})"
    )
    fig_line_episodes.update_layout(xaxis_title='Episode', yaxis_title='Score')
    fig_line_episodes.write_image(os.path.join(save_dir, f"line_episode_scores_{selected_algorithm}.png"))
    fig_line_episodes.show()

    fig_line_rewards = px.line(
        df,
        x="Episode",
        y="Reward",
        title=f"Rewards pro Episode ({selected_algorithm})"
    )
    fig_line_rewards.update_layout(xaxis_title='Episode', yaxis_title='Reward')
    fig_line_rewards.write_image(os.path.join(save_dir, f"line_episode_rewards_{selected_algorithm}.png"))
    fig_line_rewards.show()

    fig_line_cum = go.Figure()
    fig_line_cum.add_trace(go.Scatter(
        x=list(range(1, len(mean_scores) + 1)),
        y=mean_scores,
        mode='lines',
        name=f'{selected_algorithm} (Kumul. Avg)'
    ))
    fig_line_cum.update_layout(
        title=f'Kumulative Durchschnitts-Scores pro Episode ({selected_algorithm})',
        xaxis_title='Episode',
        yaxis_title='Kumulativer Durchschnitts-Score'
    )
    fig_line_cum.write_image(os.path.join(save_dir, f"line_cumulative_avg_scores_{selected_algorithm}.png"))
    fig_line_cum.show()

    fig_line_cum_reward = go.Figure()
    fig_line_cum_reward.add_trace(go.Scatter(
        x=list(range(1, len(mean_rewards) + 1)),
        y=mean_rewards,
        mode='lines',
        name=f'{selected_algorithm} (Kumul. Avg)'
    ))
    fig_line_cum_reward.update_layout(
        title=f'Kumulative Durchschnitts-Rewards pro Episode ({selected_algorithm})',
        xaxis_title='Episode',
        yaxis_title='Kumulativer Durchschnitts-Reward'
    )
    fig_line_cum_reward.write_image(os.path.join(save_dir, f"line_cumulative_avg_rewards_{selected_algorithm}.png"))
    fig_line_cum_reward.show()

    categories = ["Min Score", "Max Score"]
    values = [min_score, max_score]

    fig_min_max = go.Figure(data=[
        go.Bar(name=selected_algorithm, x=categories, y=values, marker_color='skyblue')
    ])
    fig_min_max.update_layout(
        title=f'Vergleich der minimalen und maximalen Scores ({selected_algorithm})',
        xaxis_title='Kategorie',
        yaxis_title='Score',
        barmode='group'
    )
    fig_min_max.write_image(os.path.join(save_dir, f"min_max_scores_comparison_{selected_algorithm}.png"))
    fig_min_max.show()

    categories_reward = ["Min Reward", "Max Reward"]
    values_reward = [min_reward, max_reward]

    fig_min_max_reward = go.Figure(data=[
        go.Bar(name=selected_algorithm, x=categories_reward, y=values_reward, marker_color='lightgreen')
    ])
    fig_min_max_reward.update_layout(
        title=f'Vergleich der minimalen und maximalen Rewards ({selected_algorithm})',
        xaxis_title='Kategorie',
        yaxis_title='Reward',
        barmode='group'
    )
    fig_min_max_reward.write_image(os.path.join(save_dir, f"min_max_rewards_comparison_{selected_algorithm}.png"))
    fig_min_max_reward.show()

    print(f"Alle Vergleichsmetriken für {selected_algorithm} wurden geplottet und im Verzeichnis '{save_dir}' gespeichert.")
