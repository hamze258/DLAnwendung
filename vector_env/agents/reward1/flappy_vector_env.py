import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from vector_env.src.entities import Background, Floor, Pipes, Player, Score
from  vector_env.src.utils import GameConfig, Window, Images, Sounds


class FlappyBirdEnv(gym.Env):
    """
    Gym Environment für Flappy Bird basierend auf der bestehenden Struktur.
    Nutzt src.entities und src.utils, inklusive der Pipes-Logik aus dem Repository.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        self.render_mode = render_mode
        
        if render_mode == "human":
            pygame.init()
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

        self.window = None if render_mode != "human" else self.config.window

        # Gym Spaces
        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 10, 2, 2, 2], dtype=np.float32),  # Increased upper bounds
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
        self.background.tick()
        self.score.tick()
        self.pipes.tick()
        self.floor.tick()
        self.player.tick()

        # Kollision prüfen und Belohnungen berechnen
        reward = 0.1  # Kleine Belohnung für Überleben
        if self.player.collided(self.pipes, self.floor):
            self.gameover = True
            reward = -1

        # Punkte erhöhen, wenn Pipes passiert werden
        for pipe in self.pipes.lower:
            if not pipe.scored and pipe.cx < self.player.cx:
                self.score.add()
                reward += 1.0  # Belohnung für das Passieren einer Pipe
                pipe.scored = True  # Markiere die Pipe als gezählt

        # Beobachtung erstellen
        observation = self._get_observation()

        # Episode beenden, wenn Spiel vorbei ist
        done = self.gameover
        info = {"score": self.score.score}
        # if self.gameover:
        #     print(f"Total Reward for Episode: {self.score.score}")

        self.step_count += 1

        # Rendering nach dem Update
        if self.render_mode in ["human", "rgb_array"]:
            self.render()
        
        #print(f"Player Position: ({self.player.x}, {self.player.y})")
        #print(f"Number of Pipes: Upper - {len(self.pipes.upper)}, Lower - {len(self.pipes.lower)}")
        # if len(self.pipes.upper) > 0:
        #     print(f"First Pipe Position: Upper ({self.pipes.upper[0].x}, {self.pipes.upper[0].y}), "
        #         f"Lower ({self.pipes.lower[0].x}, {self.pipes.lower[0].y})")

        
        return observation, reward, done, False, info


    def _get_observation(self):
        """Create an observation based on the current game state."""
        bird_y = self.player.y / self.config.window.viewport_height
        bird_velocity = self.player.vel_y / 10

        # Find the next pipe
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
            next_pipe_x = 2.0
            next_pipe_top_y = 1.0 
            next_pipe_bottom_y = 1.0 

        # **Ensure all observation values are within bounds**
        bird_y = np.clip(bird_y, 0.0, 1.0)
        bird_velocity = np.clip(bird_velocity, -10.0, 10.0)
        next_pipe_x = np.clip(next_pipe_x, 0.0, 2.0)
        next_pipe_top_y = np.clip(next_pipe_top_y, 0.0, 2.0)
        next_pipe_bottom_y = np.clip(next_pipe_bottom_y, 0.0, 2.0)

        observation = np.array([bird_y, bird_velocity, next_pipe_x, next_pipe_top_y, next_pipe_bottom_y], dtype=np.float32)

        return observation

    def render(self):
        """Das Spiel rendern basierend auf dem angegebenen Modus."""
        mode = self.render_mode

        if mode == "human":
        # Ereignisse verarbeiten nur im human-Modus
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        # Zeichne die Objekte
        if mode == "human" and self.window:
            self.config.screen.fill((0, 0, 0))  # Bildschirm leeren
            self.background.draw()
            self.pipes.draw()
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            pygame.display.update()
            self.config.clock.tick(self.config.fps)
            
        
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
