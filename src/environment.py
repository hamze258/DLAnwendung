import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import configs
import assets
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.score import Score


class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        pygame.init()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(2)  # 0 = keine Aktion, 1 = Fliegen
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(configs.SCREEN_HEIGHT, configs.SCREEN_WIDTH, 3), dtype=np.uint8
        )

        assets.load_sprites()
        assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        self.bird, self.score = None, None
        self.clock = pygame.time.Clock()

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        return Bird(self.sprites), Score(self.sprites)

    def reset(self, seed=None, options=None):
        """Setzt die Umgebung zurück und gibt die initiale Observation zurück."""
        super().reset(seed=seed)  # Wichtig für Gymnasium-Konformität
        if seed is not None:
            np.random.seed(seed)  # Falls Zufallszahlen im Spiel verwendet werden

        self.sprites.empty()
        self.bird, self.score = self.create_sprites()

        self.gameover = False
        self.score.value = 0

        return self._get_observation(), {}

    def step(self, action):
        if action == 1:
            self.bird.flap = 0
            self.bird.flap -= 6
        self.sprites.update()



        if self.bird.check_collision(self.sprites):
            self.gameover = True
            reward = -18
            done = True
        else:
            reward = 2
            done = False

        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                reward += 7
                assets.play_audio("point")

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        screen_data = pygame.surfarray.array3d(self.screen)
        return np.transpose(screen_data, (1, 0, 2))

    def render(self, mode='human'):
        if self.render_mode == "human":
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            pygame.event.pump()  # Pygame-Events verarbeiten
            self.clock.tick(30)
        else:
            print("finished")
            pass

    def close(self):
        pygame.quit()
