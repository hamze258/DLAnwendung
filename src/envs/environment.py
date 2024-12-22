import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import configs
import assets
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.score import Score

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        pygame.init()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(2)  # 0 = keine Aktion, 1 = Fliegen
        self.observation_space = spaces.Box(
            low=np.array([0,-1, 0, -1, -1], dtype=np.float32),
            high=np.array([1,1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        assets.load_sprites()
        assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        self.bird, self.score = None, None
        self.clock = pygame.time.Clock()

        self.step_count = 0
        self.FPS = configs.FPS
        self.column_spawn_interval = int(1.5 * self.FPS)

        #Metriken
        self.highest_score = 0  # Höchster Score bisher
        self.total_columns_passed = 0  # Alle Röhren insgesamt (für alle Episoden)
        self.episode_columns_passed = 0  # Röhren in der aktuellen Episode
        self.total_deaths = 0

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        return Bird(self.sprites), Score(self.sprites)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.sprites.empty()
        self.bird, self.score = self.create_sprites()

        self.gameover = False
        self.score.value = 0
        self.step_count = 0
        self.episode_columns_passed = 0

        return self._get_observation(), {}

    def step(self, action):
        if action == 1:
            self.bird.flap()

        self.step_count += 1

        if self.step_count % self.column_spawn_interval == 0:
            Column(self.sprites)

        self.sprites.update()

        # Kollision überprüfen
        if self.bird.check_collision(self.sprites):
            self.gameover = True
            reward = -1000
            done = True
        else:
            reward = 0.5
            done = False

        # Belohnung für das Passieren von Säulen
        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                reward += 5
                assets.play_audio("point")

        observation = self._get_observation()
        info = {
            "episode_columns_passed": self.episode_columns_passed,
            "highest_score": self.highest_score,
            "total_deaths": self.total_deaths
        }

        return observation, reward, done, False, info

    def _get_observation(self):
        bird_y = self.bird.rect.y / configs.SCREEN_HEIGHT
        bird_velocity = self.bird.velocity / 10

        next_column = None
        min_distance = float('inf')
        for sprite in self.sprites:
            if isinstance(sprite, Column):
                distance = sprite.rect.x - self.bird.rect.x
                if 0 <= distance < min_distance:
                    min_distance = distance
                    next_column = sprite

        if next_column is not None:
            next_pipe_x = (next_column.rect.x - self.bird.rect.x) / configs.SCREEN_WIDTH
            next_pipe_top_y = next_column.gap_y / configs.SCREEN_HEIGHT
            next_pipe_bottom_y = (next_column.gap_y + next_column.gap) / configs.SCREEN_HEIGHT
        else:
            next_pipe_x = 1.0
            next_pipe_top_y = 0.5
            next_pipe_bottom_y = 0.5

        gap_center_y = (next_pipe_top_y + next_pipe_bottom_y) / 2
        distance_to_gap_center = bird_y - gap_center_y

        observation = np.array([
            bird_y,                     #Vertiakle Vogel Position
            bird_velocity,              # Geschwindigkeit
            next_pipe_x,               #Horizontale Röhrenposition
            gap_center_y,              #Mitte der Lücke
            distance_to_gap_center,    #Relative position zur Lücke
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        if self.render_mode == "human":
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            pygame.event.pump()
            self.clock.tick(self.FPS)

    def calculate_reward(self):
        if self.bird.check_collision(self.sprites):
            self.gameover = True
            self.total_deaths += 1
            return -10


        reward = 0.005 #Überleben

        columns = [sprite for sprite in self.sprites if isinstance(sprite, Column)]

        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                self.total_columns_passed += 1
                self.episode_columns_passed += 1
                if self.score.value > self.highest_score:
                    self.highest_score = self.score.value  # Aktualisiere höchsten Score
                reward +=1
                #assets.play_audio("point")

        return reward

    def close(self):
        pygame.quit()