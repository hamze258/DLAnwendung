import gym
import numpy as np
from gym import spaces
from src.flappy import Flappy as FlappyBirdGame

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.game = FlappyBirdGame()
        self.game.start()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32  # Beispiel
        )
        self.action_space = spaces.Discrete(2)  # 0: Nicht springen, 1: Springen

    def reset(self):
        self.game.reset()
        return self._get_observation()

    def is_tap_event(self, action):
        # Führe die Aktion im Spiel aus
        self.game.step(action)
        obs = self._get_observation()
        reward = 0.1  # Überlebens-Belohnung für jeden Schritt

        if self.game.has_passed_pipe():
            reward += 1.0  # Belohnung für das Passieren einer Röhre

        if self.game.has_collided():
            reward -= 1.0  # Strafe für Kollision
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def _get_observation(self):
        bird_x, bird_y = self.game.bird_position()
        next_pipe_x, next_pipe_y = self.game.next_pipe_position()
        velocity = self.game.bird_velocity()

        return np.array([
            next_pipe_x - bird_x,
            next_pipe_y - bird_y,
            velocity,
        ], dtype=np.float32)

    def render(self, mode="human"):
        if mode == "human":
            self.game.render()

    def close(self):
        self.game.close()
