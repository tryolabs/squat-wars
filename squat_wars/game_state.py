from enum import Enum

GAME_TIME = 30
COUNTDOWN_TIME = 3

WINDOW_HEIGHT = 1920
WINDOW_WIDTH = 1080

CAMERA_WIDTH = 800
CAMERA_HEIGHT = 448


class GameState(Enum):
    NOT_STARTED = 0
    USER_SELECTION = 1
    COUNTDOWN = 2
    STARTED = 3
    VIDEO = 4
    ENDED = 5
    QR = 6

    def next(self):
        return GameState((self.value + 1) % len(GameState))
