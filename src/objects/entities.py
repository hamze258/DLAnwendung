from ..utils import GameConfig, get_hit_mask, pixel_collision


class Entity:
    def __init__(
        self,
        config: GameConfig,
        image: Optional[pygame.Surface] = None,
        x=0,
        y=0,
        w: int = None,
        h: int = None,
        **kwargs,
    ) -> None:
        self.config = config
        self.x = x
        self.y = y
        if w or h:
            self.w = w or config.window.ratio * h
            self.h = h or w / config.window.ratio
            self.image = pygame.transform.scale(image, (self.w, self.h))
        else:
            self.image = image
            self.w = image.get_width() if image else 0
            self.h = image.get_height() if image else 0

        self.hit_mask = get_hit_mask(image) if image else None
        self.__dict__.update(kwargs)

    def update_image(
        self, image: pygame.Surface, w: int = None, h: int = None
    ) -> None:
        self.image = image
        self.hit_mask = get_hit_mask(image)
        self.w = w or (image.get_width() if image else 0)
        self.h = h or (image.get_height() if image else 0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def collide(self, other) -> bool:
        if not self.hit_mask or not other.hit_mask:
            return self.rect.colliderect(other.rect)
        return pixel_collision(
            self.rect, other.rect, self.hit_mask, other.hit_mask
        )

    def tick(self) -> None:
        self.draw()
        rect = self.rect
        if self.config.debug:
            pygame.draw.rect(self.config.screen, (255, 0, 0), rect, 1)
            # write x and y at top of rect
            font = pygame.font.SysFont("Arial", 13, True)
            text = font.render(
                f"{self.x:.1f}, {self.y:.1f}, {self.w:.1f}, {self.h:.1f}",
                True,
                (255, 255, 255),
            )
            self.config.screen.blit(
                text,
                (
                    rect.x + rect.w / 2 - text.get_width() / 2,
                    rect.y - text.get_height(),
                ),
            )

    def draw(self) -> None:
        if self.image:
            self.config.screen.blit(self.image, self.rect)


class Background(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(
            config,
            config.images.background,
            0,
            0,
            config.window.width,
            config.window.height,
        )

class Floor(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(config, config.images.base, 0, config.window.vh)
        self.vel_x = 4
        self.x_extra = self.w - config.window.w

    def stop(self) -> None:
        self.vel_x = 0

    def draw(self) -> None:
        self.x = -((-self.x + self.vel_x) % self.x_extra)
        super().draw()

class GameOver(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(
            config=config,
            image=config.images.game_over,
            x=(config.window.width - config.images.game_over.get_width()) // 2,
            y=int(config.window.height * 0.2),
        )

class Pipe(Entity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vel_x = -5

    def draw(self) -> None:
        self.x += self.vel_x
        super().draw()


class Pipes(Entity):
    upper: List[Pipe]
    lower: List[Pipe]

    def __init__(self, config: GameConfig) -> None:
        super().__init__(config)
        self.pipe_gap = 120
        self.top = 0
        self.bottom = self.config.window.viewport_height
        self.upper = []
        self.lower = []
        self.spawn_initial_pipes()

    def tick(self) -> None:
        if self.can_spawn_pipes():
            self.spawn_new_pipes()
        self.remove_old_pipes()

        for up_pipe, low_pipe in zip(self.upper, self.lower):
            up_pipe.tick()
            low_pipe.tick()

    def stop(self) -> None:
        for pipe in self.upper + self.lower:
            pipe.vel_x = 0

    def can_spawn_pipes(self) -> bool:
        last = self.upper[-1]
        if not last:
            return True

        return self.config.window.width - (last.x + last.w) > last.w * 2.5

    def spawn_new_pipes(self):
        # add new pipe when first pipe is about to touch left of screen
        upper, lower = self.make_random_pipes()
        self.upper.append(upper)
        self.lower.append(lower)

    def remove_old_pipes(self):
        # remove first pipe if its out of the screen
        for pipe in self.upper:
            if pipe.x < -pipe.w:
                self.upper.remove(pipe)

        for pipe in self.lower:
            if pipe.x < -pipe.w:
                self.lower.remove(pipe)

    def spawn_initial_pipes(self):
        upper_1, lower_1 = self.make_random_pipes()
        upper_1.x = self.config.window.width + upper_1.w * 3
        lower_1.x = self.config.window.width + upper_1.w * 3
        self.upper.append(upper_1)
        self.lower.append(lower_1)

        upper_2, lower_2 = self.make_random_pipes()
        upper_2.x = upper_1.x + upper_1.w * 3.5
        lower_2.x = upper_1.x + upper_1.w * 3.5
        self.upper.append(upper_2)
        self.lower.append(lower_2)

    def make_random_pipes(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        base_y = self.config.window.viewport_height

        gap_y = random.randrange(0, int(base_y * 0.6 - self.pipe_gap))
        gap_y += int(base_y * 0.2)
        pipe_height = self.config.images.pipe[0].get_height()
        pipe_x = self.config.window.width + 10

        upper_pipe = Pipe(
            self.config,
            self.config.images.pipe[0],
            pipe_x,
            gap_y - pipe_height,
        )

        lower_pipe = Pipe(
            self.config,
            self.config.images.pipe[1],
            pipe_x,
            gap_y + self.pipe_gap,
        )

        return upper_pipe, lower_pipe