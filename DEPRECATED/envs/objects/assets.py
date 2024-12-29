import os
import pygame

sprites = {}
audios = {}


def load_sprites():
    path = r"img_env\data\vector_env/assets\sprites"
    for file in os.listdir(path):
        sprites[file.split('.')[0]] = pygame.image.load(os.path.join(path, file))


def get_sprite(name):
    return sprites[name]


def load_audios():
    path = r"img_env\data\vector_env/assets\audios"
    for file in os.listdir(path):
        audios[file.split('.')[0]] = pygame.mixer.Sound(os.path.join(path, file))


def play_audio(name):
    audios[name].play()