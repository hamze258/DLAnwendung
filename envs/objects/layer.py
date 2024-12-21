from enum import IntEnum, auto


class Layer(IntEnum):
    BACKGROUND = 0
    OBSTACLE = 2
    FLOOR = 1
    PLAYER = 3
    UI = 4
