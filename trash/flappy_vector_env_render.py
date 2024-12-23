import pygame
import numpy as np

from envs.vector_env.flappy_vector_env import FlappyVectorEnv

# Das "configs" aus deinem Projekt, z.B.:
import envs.configs as configs
from envs.objects import assets
from envs.objects.background import Background
from envs.objects.bird import Bird
from envs.objects.column import Column
from envs.objects.floor import Floor
from envs.objects.gameover_message import GameOverMessage
from envs.objects.gamestart_message import GameStartMessage
from envs.objects.score import Score


class FlappyVectorEnvWithRendering(FlappyVectorEnv):
    """
    Erbt von FlappyVectorEnv, rendert jedoch via Pygame 
    (Hintergrund, Boden, Vogel, Säulen etc.).
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.render_mode = render_mode

        self.gameover = False
        self.gamestarted = True  # Falls du direkt starten willst
        self.game_over_message = None  # referenz auf das Sprite

        if self.render_mode == "human":
            # Pygame initialisieren
            pygame.init()
            self.screen = pygame.display.set_mode(
                (configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Flappy Bird Game v1.0.2")

            # Optional ein Icon:
            icon_path = r"data\assets\icons\red_bird.png"
            try:
                img = pygame.image.load(icon_path)
                pygame.display.set_icon(img)
            except:
                print(f"Konnte Icon nicht laden unter: {icon_path}")

            self.clock = pygame.time.Clock()

            # Assets laden
            assets.load_sprites()
            assets.load_audios()

            # Sprite-Gruppe erstellen
            self.sprites = pygame.sprite.LayeredUpdates()

            # Sprites erzeugen
            self._create_sprites()
        else:
            self.screen = None
            self.sprites = None

    def _create_sprites(self):
        """ Erstelle alle Objekte wie im Originalscript. """
        # evtl. leeren, falls Reset
        self.sprites.empty()

        # Hintergrund
        Background(0, self.sprites)
        Background(1, self.sprites)

        # Boden
        Floor(0, self.sprites)
        Floor(1, self.sprites)

        # Vogel
        self.bird = Bird(self.sprites)

        # Optional: Startnachricht, wenn gewünscht
        if not self.gamestarted:
            self.game_start_message = GameStartMessage(self.sprites)
        else:
            self.game_start_message = None

        # Score
        self.score_sprite = Score(self.sprites)
        self.score_sprite.value = 0  # Startwert

        # Säulen anlegen: oben und unten
        self.column_top = Column(self.sprites)
        self.column_bottom = Column(self.sprites)

        # Oben gespiegelt:
        if self.column_top is not None:
            self.column_top = pygame.transform.flip(
                self.column_top.image, False, True
            )

        self.game_over_message = None

    def step(self, action):
        """
        1) Elternmethode updatet Bird und pipe_distance
        2) Pygame-Sprites positionieren/zeichnen
        3) Kollisionen überprüfen
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Hier wollen wir ggf. Säulenkollision prüfen.
        # Aber: erst Render-Positionen setzen.
        if self.render_mode == "human" and self.sprites is not None:
            # -----------------------------
            # Vogel-Position
            # -----------------------------
            if self.bird:
                # 0..500 -> 0..SCREEN_HEIGHT
                new_y = int(self.bird_position * configs.SCREEN_HEIGHT / 500.0)
                self.bird.rect.y = new_y

            # -----------------------------
            # Säulen
            # -----------------------------
            if self.column_top and self.column_bottom:
                pipe_w = self.column_top.rect.width
                max_dist = 200.0
                # Mappen: 200 -> ganz rechts, 0 -> links raus
                new_x = int(
                    (self.pipe_distance / max_dist)
                    * (configs.SCREEN_WIDTH + pipe_w)
                    - pipe_w
                )

                # Y-Mitte
                gap_size = 110
                y_center = int(self.pipe_height * configs.SCREEN_HEIGHT / 500.0)

                # Oben
                self.column_top.rect.x = new_x
                self.column_top.rect.bottom = y_center - gap_size // 2

                # Unten
                self.column_bottom.rect.x = new_x
                self.column_bottom.rect.top = y_center + gap_size // 2

            # -----------------------------
            # Score
            # -----------------------------
            # Bei pipe_distance <= 0 gibt es in der Eltern-Env +1 Reward
            # => Du könntest das hier übernehmen:
            # oder du trackst den Score in der Env. 
            # Minimales Beispiel: 
            # "reward" bei Pipe-Übergang = +1 => wir addieren es 
            # (Achtung: Reward kann auch negativ sein. Dann müsstest du filtern.)
            if reward > 0:
                self.score_sprite.value += reward

            # -----------------------------
            # Säulenkollision prüfen (optional)
            # -----------------------------
            if not terminated:
                # Wenn du NICHT in der Eltern-Env die Säulenkollision checkst,
                # kannst du es hier tun. Einfach z.B. Mask-Collision:
                columns = [self.column_top, self.column_bottom]
                if any(
                    pygame.sprite.collide_mask(self.bird, c)
                    for c in columns
                    if c is not None
                ):
                    terminated = True
                    reward -= 1.0
                    # GameOver
                    if not self.gameover:
                        self.gameover = True
                        self.game_over_message = GameOverMessage(self.sprites)
                        assets.play_audio("hit")

            # -----------------------------
            # Sprites updaten (Animationen)
            # -----------------------------
            self.sprites.update()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Bei Reset rufen wir die Elternmethode auf und 
        erstellen ggf. unsere Sprites neu.
        """
        obs, info = super().reset(seed=seed, options=options)

        self.gameover = False
        self.gamestarted = True
        if self.render_mode == "human" and self.sprites is not None:
            self._create_sprites()

        return obs, info

    def render(self):
        """Zeichnet Pygame-Sprites bei 'human'-Mode."""
        if self.render_mode == "human" and self.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.screen.fill((0, 0, 0))
            self.sprites.draw(self.screen)
            pygame.display.flip()

            self.clock.tick(configs.FPS)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        super().close()
