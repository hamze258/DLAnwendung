import random

import pygame.sprite

import envs.objects.vector_env/assets as vector_env/assets
import envs.configs as configs
from envs.objects.layer import Layer


class Column(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.OBSTACLE
        self.gap = 150

        self.sprite = vector_env/assets.get_sprite("pipe-green")
        self.sprite_rect = self.sprite.get_rect()

        self.pipe_bottom = self.sprite
        self.pipe_bottom_rect = self.pipe_bottom.get_rect(topleft=(0, self.sprite_rect.height + self.gap))

        # Definiere die vertikale Position der Lücke
        self.gap_y = random.randint(100, configs.SCREEN_HEIGHT - 100)
        
        self.pipe_top = pygame.transform.flip(self.sprite, False, True)
        self.pipe_top_rect = self.pipe_top.get_rect(topleft=(0, 0))

        self.image = pygame.surface.Surface((self.sprite_rect.width, self.sprite_rect.height * 2 + self.gap),
                                            pygame.SRCALPHA)
        self.image.blit(self.pipe_bottom, self.pipe_bottom_rect)
        self.image.blit(self.pipe_top, self.pipe_top_rect)

        sprite_floor_height = vector_env/assets.get_sprite("floor").get_rect().height
        min_y = 100
        max_y = configs.SCREEN_HEIGHT - sprite_floor_height - 100

        self.rect = self.image.get_rect(midleft=(configs.SCREEN_WIDTH, random.uniform(min_y, max_y)))
        self.mask = pygame.mask.from_surface(self.image)

        self.passed = False
        self.fully_passed = False

        super().__init__(*groups)

    def update(self):
        self.rect.x -= 2

        if self.rect.right <= 0:
            self.kill()

    def is_passed(self):
        if self.rect.x < 50 and not self.passed:
            self.passed = True
            return True
        return False

    def is_fully_passed(self, bird_rect):
        # Röhre wurde bereits passiert
        if self.passed and not self.fully_passed:
            # Prüfe, ob der Vogel die Röhre vollständig verlassen hat
            if bird_rect.left > self.rect.right:
                self.fully_passed = True
                return True
        return False
