
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from DEPRECATED.envs import configs
from DEPRECATED.envs.objects import vector_env/assets
from DEPRECATED.envs.objects.background import Background
from DEPRECATED.envs.objects.bird import Bird
from DEPRECATED.envs.objects.column import Column
from DEPRECATED.envs.objects.floor import Floor
from DEPRECATED.envs.objects.score import Score

class FlappyVectorEnv(gym.Env):
    """Flappy Bird Environment for Reinforcement Learning."""
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        pygame.init()
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
        
        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation space:
        # [bird_y, bird_velocity, next_pipe_x, next_pipe_top_y, next_pipe_bottom_y]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 10, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        vector_env/assets.load_sprites()
        vector_env/assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        self.bird = None
        self.score = None
        self.clock = pygame.time.Clock()

        self.step_count = 0
        self.FPS = configs.FPS
        self.column_spawn_interval = int(1.5 * self.FPS)

        # Metrics
        self.highest_score = 0
        self.total_columns_passed = 0
        self.episode_columns_passed = 0
        self.total_deaths = 0

    def create_sprites(self):
        """Initializes all sprites for a new episode."""
        self.sprites.empty()
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        self.bird = Bird(self.sprites)
        self.score = Score(self.sprites)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.create_sprites()

        self.gameover = False
        self.score.value = 0
        self.step_count = 0
        self.episode_columns_passed = 0

        return self._get_observation(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        if action == 1:
            self.bird.flap()

        self.step_count += 1

        # Spawn new column at intervals
        if self.step_count % self.column_spawn_interval == 0:
            Column(self.sprites)

        # Update all sprites
        self.sprites.update()

        # Handle events if rendering in human mode
        if self.render_mode == "human":
            self._handle_pygame_events()

        # Check for collisions
        done = False
        reward = 0.1  # Small reward for surviving this step

        if self.bird.check_collision(self.sprites):
            self.gameover = True
            done = True
            reward = -1
            self.total_deaths += 1
            vector_env/assets.play_audio("hit")
        else:
            # Check for passing columns
            for sprite in self.sprites:
                if isinstance(sprite, Column) and sprite.is_passed() and not sprite.passed:
                    sprite.passed = True
                    self.score.value += 1
                    self.total_columns_passed += 1
                    self.episode_columns_passed += 1
                    if self.score.value > self.highest_score:
                        self.highest_score = self.score.value
                    reward += 1  # Reward for passing a column
                    vector_env/assets.play_audio("point")

        observation = self._get_observation()
        info = {
            "episode_columns_passed": self.episode_columns_passed,
            "highest_score": self.highest_score,
            "total_deaths": self.total_deaths
        }

        return observation, reward, done, False, info

    def _get_observation(self):
        """Constructs the observation from the current game state."""
        bird_y = self.bird.rect.centery / configs.SCREEN_HEIGHT
        bird_velocity = self.bird.velocity / 10  # Normalize velocity

        # Find the next column
        next_column = None
        min_distance = float('inf')
        for sprite in self.sprites:
            if isinstance(sprite, Column):
                distance = sprite.rect.x - self.bird.rect.x
                if 0 <= distance < min_distance:
                    min_distance = distance
                    next_column = sprite

        if next_column:
            next_pipe_x = distance / configs.SCREEN_WIDTH
            next_pipe_top_y = next_column.gap_y / configs.SCREEN_HEIGHT
            next_pipe_bottom_y = (next_column.gap_y + next_column.gap) / configs.SCREEN_HEIGHT
        else:
            next_pipe_x = 1.0
            next_pipe_top_y = 0.5
            next_pipe_bottom_y = 0.5

        observation = np.array([
            bird_y,                     # Vertikale Vogelposition
            bird_velocity,             # Geschwindigkeit
            next_pipe_x,               # Horizontale Röhrenposition
            next_pipe_top_y,           # Oberes Ende der Lücke
            next_pipe_bottom_y         # Unteres Ende der Lücke
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        """Renders the environment."""
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))  # Fill with black or another background color
            self.sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(self.FPS)
        elif mode == "rgb_array":
            # Return an RGB array of the current screen
            if self.screen:
                return pygame.surfarray.array3d(self.screen)

    def _handle_pygame_events(self):
        """Handles Pygame events to prevent the window from becoming unresponsive."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        """Performs any necessary cleanup."""
        if self.render_mode == "human":
            pygame.quit()
