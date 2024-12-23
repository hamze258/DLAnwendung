import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from envs import configs
from envs.objects import assets
from envs.objects.background import Background
from envs.objects.bird import Bird
from envs.objects.column import Column
from envs.objects.floor import Floor
from envs.objects.score import Score


class FlappyVectorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyVectorEnv, self).__init__()
        self.render_mode = render_mode
        
        # -------------------- Action- und Observation-Space -------------------- #
        self.action_space = spaces.Discrete(2)  # 0 = Keine Aktion, 1 = Flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0], dtype=np.float32),
            high=np.array([500, 10, 500, 500], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.max_steps = 1000

        # -------------------- Pygame nur 1x initialisieren -------------------- #
        if not pygame.get_init():
            pygame.init()
            assets.load_sprites()
            assets.load_audios()

        # Da wir auch im Nicht-Human-Modus Sprites brauchen, legen wir sie IMMER an.
        self.screen = pygame.Surface((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
        self.sprites = pygame.sprite.LayeredUpdates()

        # Taktgeber und FPS (nur im Human-Modus nutzen wir das wirklich)
        self.clock = pygame.time.Clock()
        self.FPS = configs.FPS
        self.column_spawn_interval = int(1.5 * self.FPS)

        # -------------------- Interne Variablen -------------------- #
        self.step_count = 0
        self.gameover = False

        # Vektor-Variablen
        self.bird_position = 250
        self.bird_velocity = 0
        self.pipe_distance = 200
        self.pipe_height = 150

        self.bird = None
        self.score = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # -------------------- Grundzustände -------------------- #
        self.step_count = 0
        self.gameover = False
        self.bird_position = 250
        self.bird_velocity = 0
        self.pipe_distance = 200
        self.pipe_height = 150

        # Sprite-Gruppe leeren und neu erstellen (immer)
        self.sprites.empty()
        self.bird, self.score = self._create_sprites()
        self.score.value = 0

        # Erste Observation
        observation = np.array([
            self.bird_position,
            self.bird_velocity,
            self.pipe_distance,
            self.pipe_height
        ], dtype=np.float32)

        return observation, {}


    def _create_sprites(self):
        """Erstellt Hintergrund, Boden, Bird, Score, usw. und fügt sie der Sprite-Gruppe hinzu."""
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)

        bird = Bird(self.sprites)
        score = Score(self.sprites)
        return bird, score


    def step(self, action):
        self.step_count += 1

        # -------------------- Aktion ausführen (Flap oder nicht) -------------------- #
        if action == 1:
            self.bird.flap()
            assets.play_audio("wing")

        # Gravitation / Positionsupdate
        self.bird_velocity += 1
        self.bird_position += self.bird_velocity

        # Logische Pipe-Position
        self.pipe_distance -= 5

        # -------------------- Columns spawnen -------------------- #
        # Wir spawnen Columns IMMER, damit die Kollisionslogik auch im Nicht-Human-Modus gilt.
        if self.step_count % self.column_spawn_interval == 0:
            Column(self.sprites)

        # -------------------- Sprite-Update (inkl. Boden, Animation) -------------------- #
        # Bird-Position in Pygame-Sprite übertragen
        if self.bird is not None:
            self.bird.rect.centery = int(self.bird_position)

        # Aktualisiere alle Sprites (verschiebt Columns etc.)
        self.sprites.update()

        # -------------------- Reward-Logik -------------------- #
        # Kleiner Bonus fürs Überleben
        reward = 0.1

        # Pipe passiert? => Reset pipe_distance & Bonus
        if self.pipe_distance <= 0:
            self.pipe_distance = 200
            self.pipe_height = np.random.randint(100, 400)
            reward += 1.0
            self.score.value += 1

        terminated = False
        truncated = False

        # -------------------- Kollisionscheck (auch im Nicht-Human-Modus!) -------------------- #
        # Falls Bird None ist, kann check_collision() crashen, 
        # aber in unserem Code legen wir bird immer an => safe.
        if self.bird.check_collision(self.sprites):
            terminated = True
            self.gameover = True
            reward = -1 # Strafe für Kollision
        elif self.bird_position < 0 or self.bird_position > 500:
            terminated = True
            self.gameover = True
            reward = -1 # Strafe für Kollision

        # Zeitlimit
        if self.step_count >= self.max_steps:
            truncated = True

        # -------------------- Nächste Observation -------------------- #
        observation = np.array([
            self.bird_position,
            self.bird_velocity,
            self.pipe_distance,
            self.pipe_height
        ], dtype=np.float32)

        return observation, reward, terminated, truncated, {}


    def render(self):
        """Zeichnet die Umgebung auf den Bildschirm, aber nur, wenn render_mode=='human'."""
        if self.render_mode == "human":
            screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
            self.screen.fill((0, 0, 0))

            # Alle Sprites auf In-Memory-Surface zeichnen
            self.sprites.draw(self.screen)

            # Übertragen auf das eigentliche Fenster
            screen.blit(self.screen, (0, 0))
            pygame.display.flip()

            # Im Nicht-Human-Modus NICHT verlangsamen
            self.clock.tick(self.FPS)


    def close(self):
        """Pygame beenden (optional)."""
        if self.render_mode == "human":
            pygame.quit()
