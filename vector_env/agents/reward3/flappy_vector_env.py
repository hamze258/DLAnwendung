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

        # Gym Spaces: Jetzt 8 Dimensionen, wegen 2 Pipes
        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 10, 2, 2, 2, 2, 2, 2], dtype=np.float32),
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

        self.player.tick()
        self.pipes.tick()
        self.floor.tick()
        self.background.tick()
        self.score.tick()

        reward = 0

        # Kollision
        if self.player.collided(self.pipes, self.floor):
            self.gameover = True
            reward = -1
            observation = self._get_observation()
            return observation, reward, True, False, {"score": self.score.score}

        # Pipes passieren
        for pipe in self.pipes.lower:
            if not pipe.scored and pipe.cx < self.player.cx:
                self.score.add()
                pipe_reward = 1.0 + 0.1 * (self.score.score - 1)
                reward += pipe_reward
                pipe.scored = True

        observation = self._get_observation()
        done = self.gameover
        info = {"score": self.score.score}
        self.step_count += 1
        return observation, reward, done, False, info

    def _get_observation(self):
        bird_y = self.player.y / self.config.window.viewport_height
        bird_velocity = self.player.vel_y / 10

        # Sammle alle Pipes vor dem Vogel
        all_pipes = []
        for up, low in zip(self.pipes.upper, self.pipes.lower):
            dist = up.x - self.player.x
            if dist >= 0:
                all_pipes.append((up, low))

        # Sortiere nach x-Koordinate
        all_pipes.sort(key=lambda p: p[0].x)

        # Default-Werte
        next_pipe_x = 2.0
        next_pipe_top_y = 1.0
        next_pipe_bottom_y = 1.0

        next_next_pipe_x = 2.0
        next_next_pipe_top_y = 1.0
        next_next_pipe_bottom_y = 1.0

        # Falls mindestens ein Pipe existiert
        if len(all_pipes) >= 1:
            up1, low1 = all_pipes[0]
            next_pipe_x = (up1.x - self.player.x) / self.config.window.width
            next_pipe_top_y = up1.y / self.config.window.viewport_height
            next_pipe_bottom_y = low1.y / self.config.window.viewport_height

        # Falls mindestens zwei Pipes existieren
        if len(all_pipes) >= 2:
            up2, low2 = all_pipes[1]
            next_next_pipe_x = (up2.x - self.player.x) / self.config.window.width
            next_next_pipe_top_y = up2.y / self.config.window.viewport_height
            next_next_pipe_bottom_y = low2.y / self.config.window.viewport_height

        # Clip die Werte in ihre Grenzen
        bird_y = np.clip(bird_y, 0.0, 1.0)
        bird_velocity = np.clip(bird_velocity, -10.0, 10.0)

        next_pipe_x = np.clip(next_pipe_x, 0.0, 2.0)
        next_pipe_top_y = np.clip(next_pipe_top_y, 0.0, 2.0)
        next_pipe_bottom_y = np.clip(next_pipe_bottom_y, 0.0, 2.0)

        next_next_pipe_x = np.clip(next_next_pipe_x, 0.0, 2.0)
        next_next_pipe_top_y = np.clip(next_next_pipe_top_y, 0.0, 2.0)
        next_next_pipe_bottom_y = np.clip(next_next_pipe_bottom_y, 0.0, 2.0)

        observation = np.array([
            bird_y,
            bird_velocity,
            next_pipe_x,
            next_pipe_top_y,
            next_pipe_bottom_y,
            next_next_pipe_x,
            next_next_pipe_top_y,
            next_next_pipe_bottom_y
        ], dtype=np.float32)

        assert self.observation_space.contains(observation), f"Obs out of bounds: {observation}"
        return observation

    def render(self):
        mode = self.render_mode
        if mode == "human" and self.window:
            self.config.screen.fill((0, 0, 0))
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            pygame.display.update()
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
