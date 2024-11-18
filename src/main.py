import pygame

import assets
import configs
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.gameover_message import GameOverMessage
from objects.gamestart_message import GameStartMessage
from objects.score import Score

pygame.init()

screen = pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird Game v1.0.2")
img = pygame.image.load('assets/icons/red_bird.png')
pygame.display.set_icon(img)

clock = pygame.time.Clock()
column_create_event = pygame.USEREVENT
running = True
gameover = False
gamestarted = False

assets.load_sprites()
assets.load_audios()

sprites = pygame.sprite.LayeredUpdates()


def create_sprites():
    """Create and return initial game sprites."""
    Background(0, sprites)
    Background(1, sprites)
    Floor(0, sprites)
    Floor(1, sprites)

    return Bird(sprites), GameStartMessage(sprites), Score(sprites)


def reset_game():
    """Reset game state and all sprites."""
    global bird, game_start_message, score, gameover, gamestarted
    gameover = False
    gamestarted = False
    sprites.empty()
    bird, game_start_message, score = create_sprites()
    pygame.time.set_timer(column_create_event, 0)  # Stop column spawning initially


bird, game_start_message, score = create_sprites()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == column_create_event:
            Column(sprites)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if gameover:
                    reset_game()  # Reset the game if it was gameover
                elif not gamestarted:
                    gamestarted = True
                    game_start_message.kill()
                    pygame.time.set_timer(column_create_event, 1500)  # Start column spawning

        if not gameover:
            bird.handle_event(event)

    screen.fill(0)
    sprites.draw(screen)

    if gamestarted and not gameover:
        sprites.update()

    # Check for collision only if the game is started and not over
    if bird.check_collision(sprites) and not gameover:
        gameover = True
        gamestarted = False
        GameOverMessage(sprites)
        pygame.time.set_timer(column_create_event, 0)  # Stop column spawning
        assets.play_audio("hit")

    # Handle scoring when columns are passed
    for sprite in sprites:
        if type(sprite) is Column and sprite.is_passed():
            score.value += 1
            assets.play_audio("point")

    pygame.display.flip()
    clock.tick(configs.FPS)

pygame.quit()
