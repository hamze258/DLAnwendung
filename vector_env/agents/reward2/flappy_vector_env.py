import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from vector_env.src.entities import Background, Floor, Pipes, Player, Score
from vector_env.src.utils import GameConfig, Window, Images, Sounds


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        self.render_mode = render_mode
        
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Flappy Bird")
            self.config = GameConfig(
                screen=pygame.display.set_mode((288, 512)),
                clock=pygame.time.Clock(),
                fps=30,
                window=Window(288, 512),
                images=Images(),
                sounds=Sounds(),
            )
        else:
            self.config = create_headless_config()

        self.window = None if render_mode != "human" else self.config.window

        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        self.step_count = 0
        self.gameover = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)

        self.step_count = 0
        self.gameover = False

        return self._get_observation(), {}

    def step(self, action):

        observation = self._get_observation()

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        self.background.tick()
        self.score.tick()
        self.pipes.tick()
        self.floor.tick()
        self.player.tick()

        reward = 0

        if action == 1:
            self.player.flap()


        if self.player.collided(self.pipes, self.floor):
            self.gameover = True
            reward = -1

        # Reward for passing pipes
        for pipe in self.pipes.lower:
            if not pipe.scored and pipe.cx < self.player.cx:
                self.score.add()
                reward += 10.0
                pipe.scored = True

        # Belohnung oder Strafe basierend auf der relativen Höhe zur nächsten Pipe
        next_pipe = self._get_next_pipe()
        if next_pipe:
            bird_y = self.player.y / self.config.window.viewport_height
            pipe_mid = ((next_pipe[0].bottom_y + next_pipe[1].y) / 2) / self.config.window.viewport_height
            relative_height = bird_y - pipe_mid

            # Belohnung für Nähe zur Mitte der Pipe-Gap
            reward += (1 - abs(relative_height))
        else:
            # Hier belohnen wir den Vogel dafür, dass er eine geringe vertikale Geschwindigkeit hat
            target_velocity = 0.0
            velocity_penalty_factor = 0.5
            velocity_penalty = abs(self.player.vel_y - target_velocity) * velocity_penalty_factor
            reward -= velocity_penalty

        done = self.gameover
        info = {"score": self.score.score}

        self.step_count += 1

        return observation, reward, done, False, info

    def _get_observation(self):
        bird_y = self.player.y / self.config.window.viewport_height
        bird_velocity = self.player.vel_y / 10.0

        next_pipe = self._get_next_pipe()

        if next_pipe:
            upper_pipe, lower_pipe = next_pipe
            next_pipe_x = (upper_pipe.x - self.player.x) / self.config.window.width
            next_pipe_top_y = upper_pipe.bottom_y / self.config.window.viewport_height
            next_pipe_bottom_y = lower_pipe.y / self.config.window.viewport_height
            pipe_mid = (next_pipe_top_y + next_pipe_bottom_y) / 2.0
            relative_height = bird_y - pipe_mid

            # Berechnung der Pipe-Breite, normalisiert
            pipe_width = upper_pipe.w / self.config.window.width

            # Berechnung der Pipe-Lückengröße, normalisiert
            pipe_gap_size = (lower_pipe.y - upper_pipe.bottom_y) / self.config.window.viewport_height
        else:
            next_pipe_x = 2.0
            next_pipe_top_y = 0.2
            next_pipe_bottom_y = 0.5
            relative_height = 0.0
            pipe_width = 0.0
            pipe_gap_size = 0.3

        # Clipping der Werte
        bird_y = np.clip(bird_y, 0.0, 1.0)
        bird_velocity = np.clip(bird_velocity, -10.0, 10.0)
        next_pipe_x = np.clip(next_pipe_x, 0.0, 2.0)
        next_pipe_top_y = np.clip(next_pipe_top_y, 0.0, 1.0)
        next_pipe_bottom_y = np.clip(next_pipe_bottom_y, 0.0, 1.0)
        relative_height = np.clip(relative_height, -1.0, 1.0)
        pipe_width = np.clip(pipe_width, 0.0, 1.0)
        pipe_gap_size = np.clip(pipe_gap_size, 0.0, 2.0)

        observation = np.array([
            bird_y,
            bird_velocity,
            next_pipe_x,
            next_pipe_top_y,
            next_pipe_bottom_y,
            relative_height,
            pipe_width,
            pipe_gap_size
        ], dtype=np.float32)

        return observation


    def _get_next_pipe(self):
        next_pipe = None
        min_distance = float('inf')

        for upper_pipe, lower_pipe in zip(self.pipes.upper, self.pipes.lower):
            distance = upper_pipe.x - self.player.x
            if 0 <= distance < min_distance:
                min_distance = distance
                next_pipe = (upper_pipe, lower_pipe)

        return next_pipe

    def render(self):
        mode = self.render_mode

        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            if self.window:
                # Hintergrund löschen
                self.config.screen.fill((0, 0, 0))
                
                # Zeichne Hintergrund, Pipes, Boden und Spieler
                self.background.draw()
                self.pipes.draw()

                # Zeichne Rechtecke für die Lücken der Pipes
                next_pipe = self._get_next_pipe()
                if next_pipe:
                    upper_pipe, lower_pipe = next_pipe
                    pipe_x = upper_pipe.x
                    pipe_width = upper_pipe.w
                    pipe_gap_top = upper_pipe.bottom_y
                    pipe_gap_bottom = lower_pipe.y

                    pygame.draw.rect(
                        self.config.screen,
                        (0, 255, 0),  # Grüne Farbe
                        pygame.Rect(
                            pipe_x,
                            pipe_gap_top,
                            pipe_width,
                            pipe_gap_bottom - pipe_gap_top,
                        ),
                        2,  # Linienbreite
                    )
                
                # **Kollisionsbereiche zur Debugging-Zwecken zeichnen**
                pygame.draw.rect(
                    self.config.screen,
                    (255, 0, 0),  # Rote Farbe für Kollisionsbereiche
                    pygame.Rect(
                        self.player.x, 
                        self.player.y, 
                        self.player.w, 
                        self.player.h
                    ),
                    2,  # Linienbreite
                )
                for upper_pipe, lower_pipe in zip(self.pipes.upper, self.pipes.lower):
                    pygame.draw.rect(
                        self.config.screen,
                        (255, 0, 0),
                        pygame.Rect(
                            upper_pipe.x, 
                            upper_pipe.y, 
                            upper_pipe.w, 
                            upper_pipe.h
                        ),
                        2,
                    )
                    pygame.draw.rect(
                        self.config.screen,
                        (255, 0, 0),
                        pygame.Rect(
                            lower_pipe.x, 
                            lower_pipe.y, 
                            lower_pipe.w, 
                            lower_pipe.h
                        ),
                        2,
                    )

                # **Beobachtungswerte anzeigen**
                font = pygame.font.SysFont(None, 24)
                observation = self._get_observation()
                obs_text = f"Obs: {observation}"
                text_surface = font.render(obs_text, True, (255, 255, 255))
                self.config.screen.blit(text_surface, (10, 10))

                # Zeichne Boden, Spieler und Punkte
                self.floor.draw()
                self.player.draw()
                self.score.draw()

                # Update des Displays
                pygame.display.update()
                self.config.clock.tick(self.config.fps)
        
        elif mode == "rgb_array":
            self.config.screen.fill((0, 0, 0))
            self.background.draw()
            self.pipes.draw()

            # Zeichne Rechtecke für die Lücken der Pipes
            next_pipe = self._get_next_pipe()
            if next_pipe:
                upper_pipe, lower_pipe = next_pipe
                pipe_x = upper_pipe.x
                pipe_width = upper_pipe.w
                pipe_gap_top = upper_pipe.bottom_y
                pipe_gap_bottom = lower_pipe.y

                pygame.draw.rect(
                    self.config.screen,
                    (0, 255, 0),  # Grüne Farbe
                    pygame.Rect(
                        pipe_x,
                        pipe_gap_top,
                        pipe_width,
                        pipe_gap_bottom - pipe_gap_top,
                    ),
                    2,  # Linienbreite
                )
            self.floor.draw()
            self.player.draw()
            self.score.draw()
            return pygame.surfarray.array3d(self.config.screen).transpose((1, 0, 2))
        else:
            return None

    def close(self):
        pygame.quit()

def create_headless_config():
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    screen = pygame.Surface((288, 512))
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

# Observation Space erweitert:

# Relativer Abstand zur Mitte der Pipe (relative_height).
# Horizontaler Abstand zur nächsten Pipe.
# Reward-Funktion angepasst:

# Bestrafung bei Abweichung von der Pipe-Mitte.
# Belohnung für das Passieren von Pipes.
