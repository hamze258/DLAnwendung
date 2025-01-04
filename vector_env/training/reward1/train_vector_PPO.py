import time
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from vector_env.agents.reward1.flappy_vector_env import FlappyBirdEnv
# from stable_baselines3.common.vec_env import VecTransposeImage  # Falls gebraucht für Bildobservations


class DetailedMetricsCallback(BaseCallback):
    """
    Callback, der Metriken auf Episodenbasis sammelt und ausgibt:
      - Rewards (Summe, Min, Max, Mittelwert)
      - TD-Error (nur relevant, wenn PPO dies abspeichert)
      - Q-Values (nur relevant, wenn PPO dies abspeichert)
      - Replay Buffer Größe (bei PPO nicht vorhanden, daher ggf. ignorieren)
      - Exploration Epsilon (bei PPO nicht vorhanden, daher ggf. ignorieren)
      - Lernrate
      - Episodenlänge
      - Aktionsverteilung (falls actions in self.locals enthalten)
      - Zeit pro Episode
      - (Optional) Loss, falls manuell in self.locals["loss"] verfügbar
    """
    def __init__(self, verbose=0):
        super(DetailedMetricsCallback, self).__init__(verbose)
        # Listen zur Zwischenspeicherung für jedes Episodenende
        self.episode_rewards = []
        self.episode_td_errors = []
        self.episode_q_values = []
        self.episode_actions = []   # neu: für die Aktionsverteilung
        self.episode_losses = []    # optional: wenn du Loss-Informationen speicherst

        # Für Episodenlänge und Zeitmessung
        self.current_episode_length = 0
        self.episode_lengths = []
        self.episode_start_time = None
        self.episode_durations = []

    def _on_training_start(self) -> None:
        """
        Wird aufgerufen, bevor das Training beginnt. Setzt den Timer für die erste Episode.
        """
        self.episode_start_time = time.time()
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """
        Wird nach jedem Umgebungs-Schritt aufgerufen.
        Hier sammeln wir Step-bezogene Daten. Sobald 'done=True', loggen wir die Episode.
        """
        # 1) Rewards
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            self.episode_rewards.extend(rewards)

        # 2) TD-Error (nur wenn du es manuell in self.locals speicherst)
        if "td_error" in self.locals:
            td_error = self.locals["td_error"].detach().cpu().numpy()
            if td_error.ndim == 0:
                td_error = [td_error]
            self.episode_td_errors.extend(td_error)

        # 3) Q-Values (nur wenn du sie manuell in self.locals speicherst)
        if "q_values" in self.locals:
            q_values = self.locals["q_values"].detach().cpu().numpy()
            if q_values.ndim == 0:
                q_values = [q_values]
            self.episode_q_values.extend(q_values)

        # 4) Actions (zur Analyse der Verteilung)
        if "actions" in self.locals:
            actions = self.locals["actions"]
            self.episode_actions.extend(actions)

        # 5) (Optional) Loss
        if "loss" in self.locals:
            loss_val = self.locals["loss"]
            # Falls es ein Tensor ist, in numpy konvertieren
            if torch.is_tensor(loss_val):
                loss_val = loss_val.detach().cpu().numpy().flatten()
            elif not isinstance(loss_val, (list, np.ndarray)):
                # Falls es nur ein einzelner float ist
                loss_val = [loss_val]
            self.episode_losses.extend(loss_val)

        # 6) Episodenlänge hochzählen
        self.current_episode_length += 1

        # 7) Check, ob Episode zu Ende
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if any(dones):
                self._log_episode_metrics()

        return True

    def _log_episode_metrics(self):
        """
        Sobald eine Episode endet, loggen wir die gesammelten Metriken
        und leeren die Listen für die nächste Episode.
        """
        # 1) Episodenlänge
        self.episode_lengths.append(self.current_episode_length)

        # 2) Zeit pro Episode
        episode_end_time = time.time()
        episode_duration = episode_end_time - self.episode_start_time
        self.episode_durations.append(episode_duration)

        # --- Rewards ---
        if len(self.episode_rewards) > 0:
            reward_array = np.array(self.episode_rewards)
            self.logger.record("metrics/episode_reward_mean", np.mean(reward_array))
            self.logger.record("metrics/episode_reward_sum",  np.sum(reward_array))
            self.logger.record("metrics/episode_reward_max",  np.max(reward_array))
            self.logger.record("metrics/episode_reward_min",  np.min(reward_array))

        # --- TD-Errors ---
        if len(self.episode_td_errors) > 0:
            td_array = np.array(self.episode_td_errors)
            self.logger.record("metrics/episode_td_error_mean", np.mean(td_array))
            self.logger.record("metrics/episode_td_error_std",  np.std(td_array))
            self.logger.record("metrics/episode_td_error_max",  np.max(td_array))
            self.logger.record("metrics/episode_td_error_min",  np.min(td_array))

        # --- Q-Values ---
        if len(self.episode_q_values) > 0:
            q_array = np.array(self.episode_q_values)
            self.logger.record("metrics/episode_q_value_mean", np.mean(q_array))
            self.logger.record("metrics/episode_q_value_std",  np.std(q_array))
            self.logger.record("metrics/episode_q_value_max",  np.max(q_array))
            self.logger.record("metrics/episode_q_value_min",  np.min(q_array))

        # --- Aktionsverteilung (diskrete Actions) ---
        if len(self.episode_actions) > 0:
            actions_array = np.array(self.episode_actions)
            unique_actions, counts = np.unique(actions_array, return_counts=True)
            total_actions = len(actions_array)
            for action, count in zip(unique_actions, counts):
                self.logger.record(f"action_distribution/action_{action}", count / total_actions)

        # --- Episodenlänge (Min/Max/Mean) über alle Episoden bisher ---
        length_array = np.array(self.episode_lengths)
        self.logger.record("metrics/episode_length_mean", np.mean(length_array))
        self.logger.record("metrics/episode_length_max",  np.max(length_array))
        self.logger.record("metrics/episode_length_min",  np.min(length_array))

        # --- Zeit pro Episode (Min/Max/Mean) ---
        duration_array = np.array(self.episode_durations)
        self.logger.record("metrics/episode_time_mean", np.mean(duration_array))
        self.logger.record("metrics/episode_time_max",  np.max(duration_array))
        self.logger.record("metrics/episode_time_min",  np.min(duration_array))

        # --- Replay Buffer Größe (bei PPO nicht vorhanden, nur relevant bei Off-Policy Algorithmen) ---
        if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            replay_buffer_size = self.model.replay_buffer.size()
            self.logger.record("metrics/replay_buffer_size", replay_buffer_size)

        # --- Exploration Parameter (epsilon) (bei PPO nicht vorhanden) ---
        if hasattr(self.model, "exploration") and self.model.exploration is not None:
            current_epsilon = self.model.exploration.get("epsilon", None)
            if current_epsilon is not None:
                self.logger.record("metrics/exploration_epsilon", current_epsilon)

        # --- Lernrate ---
        if hasattr(self.model, "lr_schedule"):
            current_learning_rate = self.model.lr_schedule(self.num_timesteps)
            self.logger.record("metrics/learning_rate", current_learning_rate)

        # --- (Optional) Loss ---
        if len(self.episode_losses) > 0:
            loss_array = np.array(self.episode_losses)
            self.logger.record("metrics/loss_mean", np.mean(loss_array))
            self.logger.record("metrics/loss_std",  np.std(loss_array))

        # Zurücksetzen für nächste Episode
        self.episode_rewards.clear()
        self.episode_td_errors.clear()
        self.episode_q_values.clear()
        self.episode_actions.clear()
        self.episode_losses.clear()

        self.current_episode_length = 0
        self.episode_start_time = time.time()

    def _on_rollout_end(self):
        """
        Wird nach jedem Rollout aufgerufen.
        Bei On-Policy-Methoden (wie PPO) entspricht ein "Rollout" typischerweise n Schritten
        (z. B. n_steps in PPO), jedoch nicht unbedingt dem Episodenende.
        """
        pass


if __name__ == "__main__":
    # Aktuelles Gerät ausgeben
    if torch.cuda.is_available():
        print("Aktuelles Gerät:", torch.cuda.current_device())
        print("Gerätename:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 1. Umgebung initialisieren und überprüfen
    env = FlappyBirdEnv(render_mode="rgb_array")
    check_env(env, warn=True)

    # Train- und Eval-Environments
    vec_env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])
    eval_env = DummyVecEnv([lambda: FlappyBirdEnv(render_mode="rgb_array")])

    # Callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="vector_env/models/PPO/training4",
        log_path="vector_env/logs/PPO/training4",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    detailed_metrics_callback = DetailedMetricsCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="vector_env/models/checkpoints/PPO/training4",
        name_prefix="PPO_Flappy_Bird"
    )

    # GPU oder CPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PPO-Modell
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=1e-4,   
        n_steps=1024,         
        batch_size=64,        
        n_epochs=10,          
        gamma=0.995,           
        gae_lambda=0.95,      
        clip_range=0.1,       
        ent_coef=0.01,        
        tensorboard_log="tensorboard/PPO",
        device=device
    )

    # Training
    model.learn(
        total_timesteps=3_000_000,
        callback=[eval_callback, checkpoint_callback, detailed_metrics_callback],
        log_interval=1000,
        progress_bar=True,
    )

    # Modell speichern
    model.save("vector_env/models/PPO")
