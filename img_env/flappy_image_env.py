import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2  # OpenCV für Bildverarbeitung
from envs import configs
from envs.objects import assets
from envs.objects.background import Background
from envs.objects.bird import Bird
from envs.objects.column import Column
from envs.objects.floor import Floor
from envs.objects.score import Score



class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        pygame.init()

        self.n_frames = 4
        self.render_mode = render_mode
        self.screen = pygame.Surface((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))  # Nur in-memory Surface

        # Bild-Observation-Space: Graustufenbilder mit einer festen Größe
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 112, 4), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(2)  # 0 = keine Aktion, 1 = Fliegen

        assets.load_sprites()
        assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        self.bird, self.score = None, None
        self.clock = pygame.time.Clock()

        self.step_count = 0
        self.FPS = configs.FPS
        self.column_spawn_interval = int(1.5 * self.FPS)

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        return Bird(self.sprites), Score(self.sprites)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sprites.empty()
        self.bird, self.score = self.create_sprites()
        self.gameover = False
        self.score.value = 0
        self.step_count = 0

        initial_observation = self._get_observation()
        self.frame_buffer = np.repeat(initial_observation, self.n_frames, axis=-1)
        return self.frame_buffer, {}


    def step(self, action):
        if action == 1:
            self.bird.flap()

        self.step_count += 1
        if self.step_count % self.column_spawn_interval == 0:
            Column(self.sprites)

        self.sprites.update()
        reward = self.calculate_reward()
        done = self.gameover

        new_observation = self._get_observation()
        self.frame_buffer = np.append(self.frame_buffer[:, :, 1:], new_observation, axis=-1)

        info = {"score": self.score.value}
        return self.frame_buffer, reward, done, False, info


    def _get_observation(self):
        """Konvertiere den aktuellen Spielstatus in ein Bild im Format (64, 64, 1)."""
        # Zeichne die Umgebung auf die In-Memory-Screen-Surface
        self.screen.fill(0)
        self.sprites.draw(self.screen)

        # Hole die Pixel-Daten von der Surface
        pixels = pygame.surfarray.array3d(self.screen)  # Shape: (Width, Height, 3)

        # Transponiere und skaliere das Bild
        pixels = np.transpose(pixels, (1, 0, 2))  # Von (W, H, C) nach (H, W, C)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)  # Konvertiere zu Graustufen
        resized = cv2.resize(gray, (112, 64), interpolation=cv2.INTER_AREA)  # Größe ändern

        # Formatiere in (144, 256, 1)
        return resized[:, :, None].astype(np.uint8)

    def render(self, mode='human'):
        if self.render_mode == "human":
            screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
            screen.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)  #Synchronisiere die Framerate


    def calculate_reward(self):
        if self.bird.check_collision(self.sprites):
            self.gameover = True
            return -1000

        reward =  1 # Überleben
        

        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                reward += 10 

        return reward

    def close(self):
        pygame.quit()
