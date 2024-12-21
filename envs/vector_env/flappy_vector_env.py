import gym
from gym import spaces
import numpy as np

class FlappyVectorEnv(gym.Env):
    def __init__(self):
        super(FlappyVectorEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: No flap, 1: Flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0]),
            high=np.array([500, 10, 500, 500]),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        # Initialize the game state
        self.bird_position = 250
        self.bird_velocity = 0
        self.pipe_distance = 200
        self.pipe_height = 150
        return np.array([self.bird_position, self.bird_velocity, self.pipe_distance, self.pipe_height])

    def step(self, action):
        reward = 0
        done = False
        if action == 1:  # Flap
            self.bird_velocity = -5
        self.bird_velocity += 1
        self.bird_position += self.bird_velocity
        self.pipe_distance -= 5

        if self.pipe_distance <= 0:
            self.pipe_distance = 200
            self.pipe_height = np.random.randint(100, 400)
            reward = 1

        if self.bird_position < 0 or self.bird_position > 500:
            done = True
            reward = -1

        state = np.array([self.bird_position, self.bird_velocity, self.pipe_distance, self.pipe_height])
        return state, reward, done, {}

    def render(self, mode="human"):
        pass
