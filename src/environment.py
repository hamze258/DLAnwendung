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
            low=np.array([0, -np.inf, 0, 0], dtype=np.float32),
            high=np.array([configs.SCREEN_HEIGHT, np.inf, configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT], dtype=np.float32),
            dtype=np.float32
        )

        assets.load_sprites()
        assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        self.bird, self.score = None, None
        self.clock = pygame.time.Clock()

        # Schrittzähler und Säulenerstellungsintervall
        self.step_count = 0
        self.FPS = configs.FPS  # Stelle sicher, dass configs.FPS definiert ist
        self.column_spawn_interval = int(1.5 * self.FPS)  # Alle 1,5 Sekunden

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
        self.step_count = 0  # Schrittzähler zurücksetzen

        return self._get_observation(), {}

    def step(self, action):
        if action == 1:
            self.bird.flap()

        # Schrittzähler erhöhen
        self.step_count += 1

        # Überprüfe, ob es Zeit ist, eine neue Säule zu erstellen
        if self.step_count % self.column_spawn_interval == 0:
            Column(self.sprites)

        # Spielzustand aktualisieren
        self.sprites.update()

        # Kollision überprüfen
        if self.bird.check_collision(self.sprites):
            self.gameover = True
            reward = -1000
            done = True
        else:
            reward = 0,5
            done = False

        # Belohnung für das Passieren von Säulen
        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                reward += 5
                assets.play_audio("point")

        observation = self._get_observation()
        info = {}

        return observation, reward, done, False, info

    def _get_observation(self):
        bird_y = self.bird.rect.y
        bird_velocity = self.bird.velocity

        # Finde die nächste Säule
        next_column = None
        min_distance = float('inf')
        for sprite in self.sprites:
            if isinstance(sprite, Column):
                distance = sprite.rect.x - self.bird.rect.x
                if distance >= 0 and distance < min_distance:
                    min_distance = distance
                    next_column = sprite

        if next_column is not None:
            next_pipe_x = next_column.rect.x
            next_pipe_y = next_column.gap_y
        else:
            next_pipe_x = configs.SCREEN_WIDTH
            next_pipe_y = configs.SCREEN_HEIGHT / 2

        observation = np.array([bird_y, bird_velocity, next_pipe_x, next_pipe_y], dtype=np.float32)
        return observation

    def render(self, mode='human'):
        if self.render_mode == "human":
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            pygame.event.pump()
            self.clock.tick(self.FPS)
        else:
            pass

    def close(self):
        pygame.quit()
