import pygame.sprite

from DEPRECATED.envs.objects import vector_env/assets
import  DEPRECATED.envs.configs as configs
from  DEPRECATED.envs.objects.layer import Layer


class Background(pygame.sprite.Sprite):
    def __init__(self, index, *groups):
        self._layer = Layer.BACKGROUND
        self.image = vector_env/assets.get_sprite("background-black")
        self.rect = self.image.get_rect(topleft=(configs.SCREEN_WIDTH * index, 0))

        super().__init__(*groups)

    def update(self):
        self.rect.x -= 1

        if self.rect.right <= 0:
            self.rect.x = configs.SCREEN_WIDTH
