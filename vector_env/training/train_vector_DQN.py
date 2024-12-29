from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import torch
import numpy as np
from vector_env.agents.flappy_vector_env import FlappyBirdEnv
from stable_baselines3.common.vec_env import VecTransposeImage

# Achtung: Ggf. brauchst du VecTransposeImage nur, wenn deine Umgebung mit Bildern arbeitet.
# Falls du rein MLP und keine Bild-Daten hast, kannst du es weglassen oder nur SubprocVecEnv verwenden.

def make_env():
    """
    Hilfsfunktion, um eine Instanz deiner Umgebung zu erzeugen.
    """
    def _init():
        env = FlappyBirdEnv()
        return env
    return _init

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("Aktuelles Gerät:", torch.cuda.current_device())
        print("Gerätename:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 1. Umgebung initialisieren und überprüfen
    #    => Dafür reicht eine einzelne Instanz der Env
    single_env = FlappyBirdEnv()
    check_env(single_env, warn=True)

    # N_ENV = Anzahl paralleler Instanzen
    N_ENV = 4  # Beispielwert: 4 parallele Envs
    vec_env = SubprocVecEnv([make_env() for _ in range(N_ENV)], start_method='fork')

    # Bei Bilddaten und Convolution-Netzen kann man zusätzlich transponieren:
    # vec_env = VecTransposeImage(vec_env)

    # Für Evaluierung reicht DummyVecEnv oder ebenfalls SubprocVecEnv mit 1 Prozess
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([make_env()])

    # Benutzerdefinierter Callback für *episodenbasierte* Metriken
    class EpisodeMetricsCallback(BaseCallback):
        """
        Callback, das am Ende jeder Episode Metriken berechnet und loggt.
        """
        def __init__(self, verbose=0):
            super(EpisodeMetricsCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self) -> bool:
            """
            Wird bei jedem Schritt aufgerufen.
            Hier prüfen wir, ob eine Episode fertig ist (anhand infos).
            """
            # infos ist eine Liste von Dicts (pro Sub-Env eines).
            # Bei stable-baselines3 findest du Episodenabschluss-Infos
            # in info[i].get("terminal_observation") oder "episode".
            for info in self.locals["infos"]:
                if "episode" in info:  
                    # "episode" enthält in der Regel den episodischen Reward und die Länge
                    # z.B. info["episode"] = {"r": episodic_reward, "l": episodic_length, "t": time}
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

            return True

        def _on_rollout_end(self):
            """
            Wird aufgerufen, wenn eine Rollout-Phase (i.d.R. nach n_steps) vorbei ist,
            aber wir wollen möglicherweise schon *am Ende jeder Episode* loggen.
            Allerdings kann man hier z.B. aggregierte Werte (mean, max, etc.) loggen.
            """
            if len(self.episode_rewards) > 0:
                mean_ep_reward = np.mean(self.episode_rewards)
                max_ep_reward = np.max(self.episode_rewards)
                min_ep_reward = np.min(self.episode_rewards)
                
                mean_ep_length = np.mean(self.episode_lengths)
                max_ep_length = np.max(self.episode_lengths)
                min_ep_length = np.min(self.episode_lengths)

                # Logging
                self.logger.record("episode/mean_reward", mean_ep_reward)
                self.logger.record("episode/max_reward", max_ep_reward)
                self.logger.record("episode/min_reward", min_ep_reward)

                self.logger.record("episode/mean_length", mean_ep_length)
                self.logger.record("episode/max_length", max_ep_length)
                self.logger.record("episode/min_length", min_ep_length)

                # Zurücksetzen, damit wir nicht dauerhaft mitteln
                self.episode_rewards.clear()
                self.episode_lengths.clear()

    # 5. Callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="models/DQN/",
        log_path="vector_env/logs/DQN",
        eval_freq=10_000,  # Erhöht die Evaluationshäufigkeit für besseres Monitoring
        n_eval_episodes=10,  # Mehr Episoden für aussagekräftige Evaluation
        deterministic=True,
        render=False,
    )
    episode_metrics_callback = EpisodeMetricsCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # Weniger häufig speichern, um Platz zu sparen
        save_path="models/checkpoints/DQN/", 
        name_prefix="DQN_Flappy_Bird"
    )

    # 6. Dynamische GPU- oder CPU-Auswahl
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 7. Hyperparameter für DQN
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=5e-4,  
        buffer_size=100_000,  
        learning_starts=10_000,
        batch_size=128, 
        tau=1.0,  
        gamma=0.99,  
        train_freq=(1, "step"), 
        gradient_steps=1,
        target_update_interval=5_000, 
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        tensorboard_log="tensorboard/DQN",
        device=device,
    )

    # 9. Training
    model.learn(
        total_timesteps=3_000_000,
        callback=[eval_callback, checkpoint_callback, episode_metrics_callback],
        log_interval=1000,  # Häufigeres Logging
        progress_bar=True,  # Fortschrittsbalken
    )

    # 10. Modell speichern
    model.save("vector_env/models/DQN")
