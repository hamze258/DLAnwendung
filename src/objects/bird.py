import pygame.sprite

import assets
import configs
from layer import Layer
from objects.column import Column
from objects.floor import Floor


class Bird(pygame.sprite.Sprite):
    def __init__(self, *groups):
        super().__init__(*groups)
        self._layer = Layer.PLAYER
        self.images = [
            assets.get_sprite("redbird-upflap"),
            assets.get_sprite("redbird-midflap"),
            assets.get_sprite("redbird-downflap")
        ]

        self.image = self.images[0]
        self.rect = self.image.get_rect(topleft=(-50, 50))  # Startposition anpassen

        self.mask = pygame.mask.from_surface(self.image)
        self.velocity = 0  # Initialisiere die Geschwindigkeit des Vogels

        self.flap = 0


    def update(self):
        self.images.insert(0, self.images.pop())
        self.image = self.images[0]
        
        # Aktualisiere die Geschwindigkeit mit der Schwerkraft
        self.velocity += configs.GRAVITY
        
        # Aktualisiere die Position des Vogels
        self.rect.y += self.velocity
        #print(f"Vogelposition: x={self.rect.x}, y={self.rect.y}, velocity={self.velocity}")

        # Bewege den Vogel nach rechts, bis er die Startposition erreicht
        if self.rect.x < 50:
            self.rect.x += 3



    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.velocity = -6  # Beim Flügelschlag Geschwindigkeit nach oben setzen
            #assets.play_audio("wing")

    def check_collision(self, sprites):
        for sprite in sprites:
            if ((isinstance(sprite, Column) or isinstance(sprite, Floor)) and sprite.mask.overlap(
                self.mask,
                (self.rect.x - sprite.rect.x, self.rect.y - sprite.rect.y)
            )) or self.rect.bottom < 0:
                return True
        return False


