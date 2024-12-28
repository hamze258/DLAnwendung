import pygame

class Window:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ratio = width / height
        self.w = width
        self.h = height
        self.r = width / height
        self.viewport_width = width
        self.viewport_height = height * 0.79
        self.vw = width
        self.vh = height * 0.79
        self.viewport_ratio = self.vw / self.vh
        self.vr = self.vw / self.vh

    # Pygame Fenster erstellen
        self.screen = pygame.display.set_mode((width, height))

    def clear(self):
        """Füllt den Bildschirm mit einem Standard-Hintergrund."""
        self.screen.fill((0, 0, 0))  # Schwarzer Hintergrund (RGB: 0, 0, 0)

    def close(self):
        """Schließt das Pygame-Fenster und räumt auf."""
        pygame.quit()
