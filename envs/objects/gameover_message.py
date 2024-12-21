import pygame.sprite

import envs.objects.assets as assets
import envs.configs as configs
from envs.objects.layer import Layer


class GameOverMessage(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.UI
        self.image = assets.get_sprite("gameover")
        self.rect = self.image.get_rect(center=(configs.SCREEN_WIDTH / 2, configs.SCREEN_HEIGHT / 2))
        super().__init__(*groups)
