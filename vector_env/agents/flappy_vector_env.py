import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.entities import Background, Floor, Pipes, Player, Score
from src.utils import GameConfig, Window, Images, Sounds


class FlappyBirdEnv(gym.Env):
    """
    Gym Environment für Flappy Bird basierend auf der bestehenden Struktur.
    Nutzt src.entities und src.utils, inklusive der Pipes-Logik aus dem Repository.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        self.render_mode = render_mode
        pygame.init()
        if render_mode == "human":
            pygame.display.set_caption("Flappy Bird")
            # Normale Konfiguration mit Fenster
            self.config = GameConfig(
                screen=pygame.display.set_mode((288, 512)),
                clock=pygame.time.Clock(),
                fps=30,
                window=Window(288, 512),
                images=Images(),
                sounds=Sounds(),
            )
        else:
            # Headless-Konfiguration ohne Fenster
            self.config = create_headless_config()

        self.render_mode = render_mode
        self.window = None if render_mode != "human" else self.config.window

        # Gym Spaces
        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 10, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Spiel-Objekte
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        # Spielparameter
        self.step_count = 0
        self.gameover = False

    def reset(self, seed=None, options=None):
        """Starte eine neue Episode."""
        super().reset(seed=seed)

        # Initialisiere die Objekte neu
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        self.step_count = 0
        self.gameover = False

        return self._get_observation(), {}

    def step(self, action):
        """Einen Schritt im Spiel ausführen."""
        if action == 1:
            self.player.flap()

        # Update der Objekte
        self.player.tick()
        self.pipes.tick()
        self.floor.tick()
        

        # Debugging-Ausgaben
        print(f"Step: {self.step_count}")
        print(f"Score: {self.score.score}")
        print(f"Player Position: ({self.player.x}, {self.player.y})")
        print(f"Number of Pipes: Upper - {len(self.pipes.upper)}, Lower - {len(self.pipes.lower)}")
        if len(self.pipes.upper) > 0:
            print(f"First Pipe Position: Upper ({self.pipes.upper[0].x}, {self.pipes.upper[0].y}), "
                f"Lower ({self.pipes.lower[0].x}, {self.pipes.lower[0].y})")

        # Kollision prüfen
        reward = 0.1  # Kleine Belohnung für Überleben
        if self.player.collided(self.pipes, self.floor):
            print("Kollision erkannt!")
            self.gameover = True
            reward = -1

        # Punkte erhöhen, wenn Pipes passiert werden
        for pipe in self.pipes.lower:
            if not pipe.scored and pipe.cx < self.player.cx:
                print("Pipe passiert!")
                self.score.add()
                reward += 1.0  # Belohnung für das Passieren einer Pipe
                pipe.scored = True  # Markiere die Pipe als gezählt

        # Beobachtung erstellen
        observation = self._get_observation()

        # Episode beenden, wenn Spiel vorbei ist
        done = self.gameover
        info = {"score": self.score.score}

        self.step_count += 1
        return observation, reward, done, False, info

    def _get_observation(self):
        """Beobachtung basierend auf dem Spielzustand erstellen."""
        bird_y = self.player.y / self.config.window.viewport_height
        bird_velocity = self.player.vel_y / 10

        # Finde die nächste Pipe
        next_pipe = None
        min_distance = float('inf')

        for upper_pipe, lower_pipe in zip(self.pipes.upper, self.pipes.lower):
            distance = upper_pipe.x - self.player.x
            if 0 <= distance < min_distance:
                min_distance = distance
                next_pipe = (upper_pipe, lower_pipe)

        if next_pipe:
            upper_pipe, lower_pipe = next_pipe
            next_pipe_x = (upper_pipe.x - self.player.x) / self.config.window.width
            next_pipe_top_y = upper_pipe.y / self.config.window.viewport_height
            next_pipe_bottom_y = lower_pipe.y / self.config.window.viewport_height
        else:
            next_pipe_x = 1.0
            next_pipe_top_y = 0.5
            next_pipe_bottom_y = 0.5

        return np.array([bird_y, bird_velocity, next_pipe_x, next_pipe_top_y, next_pipe_bottom_y], dtype=np.float32)

    def render(self):
        """Das Spiel rendern basierend auf dem angegebenen Modus."""
        mode=self.render_mode
        
        # Aktualisiere alle Objekte
        self.background.tick()
        self.pipes.tick()
        self.floor.tick()
        self.player.tick()
        self.score.tick()
        
        # Zeichne die Objekte
        if mode == "human" and self.window:
            self.config.screen.fill((0, 0, 0))  # Bildschirm leeren
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            pygame.display.update()
        
        elif mode == "rgb_array":
            # Zeichnen für den 'rgb_array'-Modus
            self.config.screen.fill((0, 0, 0))
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            # Extrahiere den Frame als numpy-Array
            return pygame.surfarray.array3d(self.config.screen).transpose((1, 0, 2))
        else:
            return None

    def close(self):
        """Schließt Ressourcen."""
        pygame.quit()

def create_headless_config():
    """
    Erstellt eine GameConfig für den headless-Modus (ohne grafische Darstellung).
    """
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Dummy-Videotreiber für Headless-Betrieb
    pygame.init()
    
    screen = pygame.Surface((288, 512))  # Dummy-Screen
    clock = pygame.time.Clock()
    fps = 30
    window = Window(288, 512)
    images = Images()
    sounds = Sounds()

    return GameConfig(
        screen=screen,
        clock=clock,
        fps=fps,
        window=window,
        images=images,
        sounds=sounds,

    )
