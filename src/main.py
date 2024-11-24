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
img = pygame.image.load(r'src\assets\icons\red_bird.png')
pygame.display.set_icon(img)

clock = pygame.time.Clock()
running = True
gameover = False
gamestarted = False

assets.load_sprites()
assets.load_audios()

sprites = pygame.sprite.LayeredUpdates()

# Schrittzähler und Säulenerstellungsintervall
step_count = 0
FPS = configs.FPS
column_spawn_interval = int(1.5 * FPS)  # Alle 1,5 Sekunden

def create_sprites():
    """Create and return initial game sprites."""
    Background(0, sprites)
    Background(1, sprites)
    Floor(0, sprites)
    Floor(1, sprites)

    return Bird(sprites), GameStartMessage(sprites), Score(sprites)


def reset_game():
    """Reset game state and all sprites."""
    global bird, game_start_message, score, gameover, gamestarted, step_count
    gameover = False
    gamestarted = False
    step_count = 0  # Schrittzähler zurücksetzen
    sprites.empty()
    bird, game_start_message, score = create_sprites()


bird, game_start_message, score = create_sprites()

while running:
    step_count += 1  # Schrittzähler erhöhen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if gameover:
                    reset_game()  # Reset the game if it was gameover
                elif not gamestarted:
                    gamestarted = True
                    game_start_message.kill()
                bird.handle_event(event)

    if gamestarted and not gameover:
        # Erstelle Säulen basierend auf dem Schrittzähler
        if step_count % column_spawn_interval == 0:
            Column(sprites)

        sprites.update()

    screen.fill(0)
    sprites.draw(screen)

    # Check for collision only if the game is started and not over
    if gamestarted and not gameover:
        if bird.check_collision(sprites):
            gameover = True
            gamestarted = False
            GameOverMessage(sprites)
            assets.play_audio("hit")

    # Handle scoring when columns are passed
    for sprite in sprites:
        if isinstance(sprite, Column) and sprite.is_passed():
            score.value += 1
            assets.play_audio("point")

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
