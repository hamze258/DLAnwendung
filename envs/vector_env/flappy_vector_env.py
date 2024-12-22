import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FlappyVectorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # Action Space: 0 = No flap, 1 = Flap
        self.action_space = spaces.Discrete(2)
        self.render_mode=render_mode
        
        # Beobachtungen: [bird_position, bird_velocity, pipe_distance, pipe_height]
        # Shape sollte (4,) sein, dtype float32
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0], dtype=np.float32),
            high=np.array([500, 10, 500, 500], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # Optionale Variablen, z.B. Schrittzähler
        self.step_count = 0
        self.max_steps = 1000  # optional: Episodenlimit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Interner Zustand initialisieren
        self.step_count = 0
        self.bird_position = 250
        self.bird_velocity = 0
        self.pipe_distance = 200
        self.pipe_height = 150

        # Rückgabe: (observation, info)
        observation = np.array([
            self.bird_position, 
            self.bird_velocity, 
            self.pipe_distance, 
            self.pipe_height
        ], dtype=np.float32)

        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        # Aktion ausführen
        if action == 1:  
            self.bird_velocity = -5
        self.bird_velocity += 1
        self.bird_position += self.bird_velocity
        self.pipe_distance -= 5

        # Reward-Logik
        reward = 0.0

        # Termination (Spiel verloren)
        terminated = False
        truncated = False

        # Punkte, wenn Pipe vorbei
        if self.pipe_distance <= 0:
            self.pipe_distance = 200
            self.pipe_height = np.random.randint(100, 400)
            reward += 1

        # If Bird geht ausserhalb des Fensters
        if self.bird_position < 0 or self.bird_position > 500:
            terminated = True
            reward -= 1

        # Optionales Zeitlimit pro Episode
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Beobachtung neu berechnen
        observation = np.array([
            self.bird_position, 
            self.bird_velocity, 
            self.pipe_distance, 
            self.pipe_height
        ], dtype=np.float32)

        info = {}

        # Rückgabe: (obs, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def render(self):
        pass
