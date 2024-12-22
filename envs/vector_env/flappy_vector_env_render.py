# flappy_vector_env_render.py

import pygame
import numpy as np
import cv2
from gymnasium import spaces
from flappy_vector_env import FlappyVectorEnv

# Falls du deine Objekte (Bird, Column, Background, etc.) bereits in 'envs.objects' hast,
# kannst du diese hier importieren:
from envs.objects.background import Background
from envs.objects.bird import Bird
from envs.objects.column import Column
from envs.objects.floor import Floor
from envs.objects.score import Score
# usw.

# Oder du machst ein Minimalbeispiel mit eigenen "Sprite"-Klassen.

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

class FlappyVectorEnvWithRendering(FlappyVectorEnv):
    def __init__(self, render_mode=None):
        """
        Erbt von FlappyVectorEnv und erweitert das Rendern via Pygame.
        """
        super().__init__(render_mode=render_mode)

        # Nur wenn render_mode == 'human' (oder 'rgb_array', etc.), 
        # wird Pygame initialisiert.
        if self.render_mode == "human":
            pygame.init()
            # Hier kannst du dein Fenster erstellen
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Vector Env")
            self.clock = pygame.time.Clock()

            # Deine Sprites initialisieren
            self.sprites = pygame.sprite.LayeredUpdates()
            self._create_sprites()
        else:
            self.screen = None
            self.sprites = None

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        return Bird(self.sprites), Score(self.sprites)

    def step(self, action):
        """
        Wir überschreiben die step()-Methode, 
        behalten aber die Logik der Elternklasse bei.
        """
        # Rufe die Original-Logik des Elternteils auf
        obs, reward, terminated, truncated, info = super().step(action)

        if self.render_mode == "human":
            # Hier könntest du optional in jedem Step
            # die Pygame-Sprites aktualisieren, 
            # indem du deren Positionen an die 
            # self.bird_position, self.pipe_distance etc. anpasst.
            self._update_sprites()

        return obs, reward, terminated, truncated, info

    def _update_sprites(self):
        """
        Mache hier z.B.:
        - self.bird_sprite.rect.y = self.bird_position
        - self.pipe_sprite.rect.x = ...
        usw.
        """
        if not self.sprites:
            return
        self.bird_sprite.rect.centery = self.bird_position
        self.sprites.update()
        pass

    def render(self):
        """
        Überschreiben der render()-Methode für 'human'-Modus.
        """
        if self.render_mode == "human" and self.screen is not None:
            # Bildschirm leeren
            self.screen.fill((0, 0, 0))

            # Sprites zeichnen
            self.sprites.draw(self.screen)

            # Aktualisiere das Fenster
            pygame.display.flip()
            self.clock.tick(30)  # oder 60 FPS, je nach Bedarf

        # Optional kann man hier auch ein 'rgb_array' zurückgeben:
        # elif self.render_mode == "rgb_array":
        #     # Surface in Array konvertieren
        #     pixels = pygame.surfarray.array3d(self.screen)
        #     # (W, H, C) -> (H, W, C)
        #     pixels = np.transpose(pixels, (1, 0, 2))
        #     return pixels

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        super().close()
