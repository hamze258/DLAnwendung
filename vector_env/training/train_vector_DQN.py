from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import torch
import numpy as np
from vector_env.agents.flappy_vector_env import FlappyBirdEnv
from stable_baselines3.common.vec_env import VecTransposeImage



if __name__ == "__main__":

    if torch.cuda.is_available():
        print("Aktuelles Gerät:", torch.cuda.current_device())
        print("Gerätename:", torch.cuda.get_device_name(torch.cuda.current_device()))
    # 1. Umgebung initialisieren und überprüfen
    env = FlappyBirdEnv()
    check_env(env, warn=True)

# Train and Eval Environments müssen konsistent sein
    vec_env = DummyVecEnv([lambda: FlappyBirdEnv()])
    eval_env = DummyVecEnv([lambda: FlappyBirdEnv()])

    # Benutzerdefinierter Callback für detailliertes Logging
    class DetailedMetricsCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(DetailedMetricsCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Belohnungsmetriken
            if "rewards" in self.locals:
                rewards = self.locals["rewards"]
                self.logger.record("metrics/reward_mean", np.mean(rewards))
                self.logger.record("metrics/reward_sum", np.sum(rewards))
                self.logger.record("metrics/reward_max", np.max(rewards))
                self.logger.record("metrics/reward_min", np.min(rewards))
            
            # TD-Fehler
            if "td_error" in self.locals:
                td_error = self.locals["td_error"].detach().cpu().numpy()
                self.logger.record("metrics/td_error_mean", np.mean(td_error))
                self.logger.record("metrics/td_error_std", np.std(td_error))
                self.logger.record("metrics/td_error_max", np.max(td_error))
                self.logger.record("metrics/td_error_min", np.min(td_error))
            
            # Q-Werte
            if "q_values" in self.locals:
                q_values = self.locals["q_values"].detach().cpu().numpy()
                self.logger.record("metrics/q_value_mean", np.mean(q_values))
                self.logger.record("metrics/q_value_std", np.std(q_values))
                self.logger.record("metrics/q_value_max", np.max(q_values))
                self.logger.record("metrics/q_value_min", np.min(q_values))

            # Replay Buffer Größe
            if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                replay_buffer_size = self.model.replay_buffer.size()
                self.logger.record("metrics/replay_buffer_size", replay_buffer_size)

            # Exploration Parameter
            if hasattr(self.model, "exploration") and self.model.exploration is not None:
                current_epsilon = self.model.exploration.get("epsilon", None)
                if current_epsilon is not None:
                    self.logger.record("metrics/exploration_epsilon", current_epsilon)

            # Lernrate überwachen
            if hasattr(self.model, "lr_schedule"):
                current_learning_rate = self.model.lr_schedule(self.num_timesteps)
                self.logger.record("metrics/learning_rate", current_learning_rate)

            return True


    # 5. Callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="models/DQN/",
        log_path="vector_env\logs\DQN",
        eval_freq=10000,  # Erhöht die Evaluationshäufigkeit für besseres Monitoring
        n_eval_episodes=10,  # Mehr Episoden für aussagekräftige Evaluation
        deterministic=True,
        render=False,
    )
    detailed_metrics_callback = DetailedMetricsCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Weniger häufig speichern, um Platz zu sparen
        save_path="models/checkpoints/DQN/", 
        name_prefix="DQN_Flappy_Bird"
    )

    # 6. Dynamische GPU- oder CPU-Auswahl
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 7. Hyperparameter für DQN optimieren
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=5e-4,  
        buffer_size=100000,  
        learning_starts=10000,
        batch_size=128, 
        tau=1.0,  
        gamma=0.99,  
        train_freq=(1, "step"), 
        gradient_steps=1,
        target_update_interval=5000, 
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        tensorboard_log="tensorboard/DQN",
        device=device,
    )


    # 9. Training mit optimierten Parametern
    model.learn(
        total_timesteps=3000000,
        callback=[eval_callback, checkpoint_callback,detailed_metrics_callback],
        log_interval=1000,  # Häufigeres Logging
        progress_bar=True,  # Fortschrittsbalken für bessere Übersicht
    )

    # 10. Modell speichern
    model.save("vector_env\models\DQN")
