import asyncio
import sys

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window


class Flappy:
    def __init__(self, render=True):
        pygame.init()
        self.render = render

        # Initialisiere immer ein Fensterobjekt, auch wenn es nicht angezeigt wird
        self.window = Window(288, 512)

        if self.render:
            pygame.display.set_caption("Flappy Bird")
            window = Window(288, 512)
            screen = pygame.display.set_mode((window.width, window.height))
        else:
            # Dummy-Display für headless Betrieb
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            screen = None

        images = Images()

        self.config = GameConfig(
            screen=screen if self.render else None,  # Kein Bildschirm bei deaktiviertem Rendering
            clock=pygame.time.Clock(),
            fps=30,
            window=self.window,
            images=images,
            sounds=Sounds(),
        )

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            if self.render:
                pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                print(f"Game Over! Score: {self.score}")
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    print(f"Pipe crossed! Current score: {self.score}")
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            self.frame_count += 1
            if self.frame_count % 100 == 0:  # Alle 100 Frames den Fortschritt ausgeben
                print(f"Frames: {self.frame_count}, Score: {self.score}")

            #rendering bei spielen oder nicht
            if self.render:
                pygame.display.update()

            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            if self.render:
                pygame.display.update()
            await asyncio.sleep(0)
