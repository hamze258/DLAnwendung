import pygame.sprite

import  DEPRECATED.envs.objects.vector_env/assets as vector_env/assets
import  DEPRECATED.envs.configs as configs
from  DEPRECATED.envs.objects.layer import Layer


class Floor(pygame.sprite.Sprite):
    def __init__(self, index, *groups):
        self._layer = Layer.FLOOR
        self.image = vector_env/assets.get_sprite("floor")
        self.rect = self.image.get_rect(bottomleft=(configs.SCREEN_WIDTH * index, configs.SCREEN_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)

    def update(self):
        self.rect.x -= 2

        if self.rect.right <= 0:
            self.rect.x = configs.SCREEN_WIDTH