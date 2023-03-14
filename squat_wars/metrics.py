import time

FRAME_WINDOW = 10


class FPS:
    def __init__(self) -> None:
        self._reset()

    def start(self):
        self._reset()
        self.frame_start_time = time.time()

    def step(self):
        self.frame_count += 1
        if self.frame_count % FRAME_WINDOW == 0:
            self.fps = FRAME_WINDOW / (time.time() - self.frame_start_time)
            self.frame_start_time = time.time()

    def _reset(self):
        self.fps = 0
        self.frame_count = 0
