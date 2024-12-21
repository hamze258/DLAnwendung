import gym
from gym import spaces
import numpy as np
import cv2

class FlappyImageEnv(gym.Env):
    def __init__(self):
        super(FlappyImageEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: No flap, 1: Flap
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.game_state = np.zeros((500, 500, 3), dtype=np.uint8)
        self.bird_position = 250
        self.pipe_distance = 200
        self.pipe_height = 150
        return self._get_frame()

    def step(self, action):
        reward = 0
        done = False
        if action == 1:  # Flap
            self.bird_position -= 20
        self.bird_position += 5
        self.pipe_distance -= 5

        if self.pipe_distance <= 0:
            self.pipe_distance = 200
            self.pipe_height = np.random.randint(100, 400)
            reward = 1

        if self.bird_position < 0 or self.bird_position > 500:
            done = True
            reward = -1

        return self._get_frame(), reward, done, {}

    def _get_frame(self):
        self.game_state.fill(0)
        cv2.rectangle(self.game_state, (self.pipe_distance, self.pipe_height), (self.pipe_distance + 50, 500), (0, 255, 0), -1)
        cv2.circle(self.game_state, (250, self.bird_position), 15, (255, 0, 0), -1)
        resized = cv2.resize(cv2.cvtColor(self.game_state, cv2.COLOR_BGR2GRAY), (84, 84))
        return np.expand_dims(resized, axis=-1)

    def render(self, mode="human"):
        cv2.imshow("Flappy Bird", self.game_state)
        cv2.waitKey(1)
