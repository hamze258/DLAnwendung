import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from vector_env.src.entities import Background, Floor, Pipes, Player, Score
from vector_env.src.utils import GameConfig, Window, Images, Sounds


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        self.render_mode = render_mode
        
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Flappy Bird")
            self.config = GameConfig(
                screen=pygame.display.set_mode((288, 512)),
                clock=pygame.time.Clock(),
                fps=30,
                window=Window(288, 512),
                images=Images(),
                sounds=Sounds(),
            )
        else:
            self.config = create_headless_config()

        self.window = None if render_mode != "human" else self.config.window

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 10, 2, 2, 2, 1], dtype=np.float32),  # Added relative height
            dtype=np.float32
        )

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        self.step_count = 0
        self.gameover = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        self.step_count = 0
        self.gameover = False

        return self._get_observation(), {}

    def step(self, action):
        if action == 1:
            self.player.flap()

        self.background.tick()
        self.score.tick()
        self.pipes.tick()
        self.floor.tick()
        self.player.tick()

        reward = 0.1
        if self.player.collided(self.pipes, self.floor):
            self.gameover = True
            reward = -1

        # Reward for passing pipes
        for pipe in self.pipes.lower:
            if not pipe.scored and pipe.cx < self.player.cx:
                self.score.add()
                reward += 1.0
                pipe.scored = True

        # Reward for maintaining position near pipe center
        next_pipe = self._get_next_pipe()
        if next_pipe:
            pipe_mid = (next_pipe[0].y + next_pipe[1].y) / 2 / self.config.window.viewport_height
            relative_height = self.player.y / self.config.window.viewport_height - pipe_mid
            reward -= abs(relative_height)  # Penalize deviation from the center

        observation = self._get_observation()
        done = self.gameover
        info = {"score": self.score.score}

        self.step_count += 1

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return observation, reward, done, False, info

    def _get_observation(self):
        bird_y = self.player.y / self.config.window.viewport_height
        bird_velocity = self.player.vel_y / 10

        next_pipe = self._get_next_pipe()

        if next_pipe:
            upper_pipe, lower_pipe = next_pipe
            next_pipe_x = (upper_pipe.x - self.player.x) / self.config.window.width
            next_pipe_top_y = upper_pipe.y / self.config.window.viewport_height
            next_pipe_bottom_y = lower_pipe.y / self.config.window.viewport_height
            pipe_mid = (next_pipe_top_y + next_pipe_bottom_y) / 2
            relative_height = bird_y - pipe_mid
        else:
            next_pipe_x = 2.0
            next_pipe_top_y = 1.0
            next_pipe_bottom_y = 1.0
            relative_height = 0.0

        bird_y = np.clip(bird_y, 0.0, 1.0)
        bird_velocity = np.clip(bird_velocity, -10.0, 10.0)
        next_pipe_x = np.clip(next_pipe_x, 0.0, 2.0)
        next_pipe_top_y = np.clip(next_pipe_top_y, 0.0, 2.0)
        next_pipe_bottom_y = np.clip(next_pipe_bottom_y, 0.0, 2.0)
        relative_height = np.clip(relative_height, -1.0, 1.0)

        observation = np.array([bird_y, bird_velocity, next_pipe_x, next_pipe_top_y, next_pipe_bottom_y, relative_height], dtype=np.float32)

        return observation

    def _get_next_pipe(self):
        next_pipe = None
        min_distance = float('inf')

        for upper_pipe, lower_pipe in zip(self.pipes.upper, self.pipes.lower):
            distance = upper_pipe.x - self.player.x
            if 0 <= distance < min_distance:
                min_distance = distance
                next_pipe = (upper_pipe, lower_pipe)

        return next_pipe

    def render(self):
        mode = self.render_mode

        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        if mode == "human" and self.window:
            self.config.screen.fill((0, 0, 0))
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            pygame.display.update()
            self.config.clock.tick(self.config.fps)
        elif mode == "rgb_array":
            self.config.screen.fill((0, 0, 0))
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            return pygame.surfarray.array3d(self.config.screen).transpose((1, 0, 2))
        else:
            return None

    def close(self):
        pygame.quit()

def create_headless_config():
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    screen = pygame.Surface((288, 512))
    clock = pygame.time.Clock()
    fps = 30
    window = Window(288, 512)
    images = Images()
    sounds = Sounds()

    return GameConfig(
        screen=screen,
        clock=clock,
        fps=fps,
        window=window,
        images=images,
        sounds=sounds,
    )

# Observation Space erweitert:

# Relativer Abstand zur Mitte der Pipe (relative_height).
# Horizontaler Abstand zur nächsten Pipe.
# Reward-Funktion angepasst:

# Bestrafung bei Abweichung von der Pipe-Mitte.
# Belohnung für das Passieren von Pipes.
