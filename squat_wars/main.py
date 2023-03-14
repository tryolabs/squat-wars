import argparse
import sys
import time
from multiprocessing import Queue
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from pynput import keyboard

from squat_wars.classifier import Classifier
from squat_wars.draw_engine import DrawEngine
from squat_wars.game_state import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    COUNTDOWN_TIME,
    GAME_TIME,
    GameState,
)
from squat_wars.metrics import FPS
from squat_wars.movenet import Movenet
from squat_wars.player import Player
from squat_wars.ranking import Ranking
from squat_wars.video_writer import VideoWriter


class SquatGame:
    def __init__(self, model_path: str, camera_id: Union[int, str]) -> None:
        """
        Initialize different components such as the model, classifier, keyboard, etc
        for the game to work correctly
        """

        # type check camera_id
        if not isinstance(camera_id, (int, str)):
            raise ValueError("camera_id must be an integer or string.")

        # convert camera_id to int
        camera_id = int(camera_id) if isinstance(camera_id, str) else camera_id

        # init camera
        self.cap = cv2.VideoCapture(camera_id)

        # init window
        self.win_name = "Squat Wars"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        # comment the line below if working on macos
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            print(
                "Camera does not support specified resolution or camera backend does not support"
                " this set up"
            )
            raise RuntimeError
        print(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # init the pose estimator selected
        if Path(model_path).stem.startswith(("movenet_lightning", "movenet_thunder")):
            self.pose_detector = Movenet(model_path)
        else:
            sys.exit("ERROR: Model is not supported.")

        # init classifier
        self.classifier = Classifier(inertia=0)

        # init game state
        self.player = Player("Default")
        self.game_state = GameState.NOT_STARTED

        # event detection
        self.listener = keyboard.Listener(on_press=self.handle_event)
        self.listener.start()

        # create fps counter
        self.fps_counter = FPS()

        # time-measure variables (time measurements done in seconds)
        self.game_start_time = time.time()
        self.countdown_start_time = 0

        # init ranking
        self.ranking = Ranking.load_from_csv()
        print(f"Succesfully loaded ranking: {self.ranking}")

        # init draw engine
        self.draw = DrawEngine(self.classifier, self.ranking, self.fps_counter)

        self.frames_queue = Queue()
        self.qr_queue = Queue()
        self.video_writer = VideoWriter(self.frames_queue, self.qr_queue)

        # init variables
        self.reset()

    def handle_event(self, key: Union[keyboard.Key, keyboard.KeyCode, None]):
        """Handle keyboard events to control the flow of the game"""
        if key == keyboard.Key.esc:
            return False

        if self.game_state in [GameState.NOT_STARTED, GameState.QR] and key == keyboard.Key.ctrl_l:
            self.game_state = self.game_state.next()

        if self.game_state == GameState.USER_SELECTION:
            if isinstance(key, keyboard.KeyCode) and key.char:
                self.inputs[self.current].append(key.char)
            else:
                if key == keyboard.Key.alt_l:
                    self.game_state = GameState.NOT_STARTED

                if key == keyboard.Key.tab:
                    self.current = (self.current + 1) % 2

                if key == keyboard.Key.space:
                    self.inputs[self.current].append(" ")

                if key == keyboard.Key.enter:
                    self.countdown_start_time = time.time()
                    self.game_state = self.game_state.next()

                if key == keyboard.Key.backspace and self.inputs[self.current]:
                    self.inputs[self.current].pop()

    def prepare_frame(self):
        """Read frame from camera and transpose it, if not available return black screen"""
        ret, image = self.cap.read()
        if ret:
            return cv2.transpose(image)

        return np.zeros((CAMERA_WIDTH, CAMERA_HEIGHT, 3), dtype=np.uint8)

    def draw_screen(self, image: np.ndarray):
        """Draw image on screen"""
        cv2.imshow(self.win_name, image)
        cv2.waitKey(1)

    def reset(self):
        """Reset state variables such as the video frames, text inputs and classifier"""
        self.classifier.reset()
        self.video_frames = []
        self.inputs = {0: [], 1: []}
        self.current = 0

        # loading screen variables
        self.current_icon = 0
        self.current_msg = 0
        self.loading_time = 0

    def run(self):
        """Game loop that goes through the different game states until stopped"""
        self.fps_counter.start()
        self.video_writer.start()
        while True:
            frame = self.prepare_frame()

            ## Game states
            if self.game_state == GameState.NOT_STARTED:
                self.reset()
                person = self.pose_detector.detect(frame)
                image = self.draw.draw_person(image=frame, person=person)
                image = self.draw.not_started_screen(image)

            elif self.game_state == GameState.USER_SELECTION:
                cursor_player = "_" if self.current == 0 else ""
                cursor_email = "_" if self.current == 1 else ""
                image = self.draw.user_selection_screen(
                    frame,
                    "".join(self.inputs[0]) + cursor_player,
                    "".join(self.inputs[1]) + cursor_email,
                )
            elif self.game_state == GameState.COUNTDOWN:
                player_name = "".join(self.inputs[0]) if self.inputs[0] else "Default"
                player_email = "".join(self.inputs[1]) if self.inputs[1] else ""
                self.player = Player(name=player_name, email=player_email)

                # elapsed countdown time
                elapsed_countdown_time = time.time() - self.countdown_start_time

                # remaining countdown time update
                remaining_countdown_time = int(COUNTDOWN_TIME - elapsed_countdown_time) + 1

                # time finished
                if elapsed_countdown_time >= COUNTDOWN_TIME:
                    # preparations for STARTED state
                    self.game_state = self.game_state.next()
                    self.game_start_time = time.time()

                # draw on image
                image = self.draw.countdown(frame, remaining_countdown_time)

            elif self.game_state == GameState.STARTED:
                # detect person and classify
                person = self.pose_detector.detect(frame)

                if person:
                    self.classifier.classify(person, count=True)

                # elapsed game time
                elapsed_game_time = time.time() - self.game_start_time

                # remaining game time update
                remaining_game_time = GAME_TIME - int(elapsed_game_time)

                image = self.draw.draw_person(image=frame, person=person)
                image = self.draw.started_screen(image, self.player, remaining_game_time)

                if remaining_game_time <= 10:
                    self.video_frames.append(image)

                if elapsed_game_time > GAME_TIME:
                    self.ranking.add(self.player, self.classifier.squat_count)
                    self.ranking.save_to_csv()
                    self.game_state = self.game_state.next()

            elif self.game_state == GameState.VIDEO:
                image = self.draw.ending_screen(self.player)
                self.video_frames.append(image)
                for x in self.video_frames:
                    self.frames_queue.put(x)
                self.frames_queue.put(self.player)
                self.game_state = self.game_state.next()

            elif self.game_state == GameState.ENDED:
                try:
                    self.qr = self.qr_queue.get(timeout=0.125)
                    self.game_state = self.game_state.next()
                except Exception:
                    self.loading_time += 0.125
                    self.current_icon = (self.current_icon + 1) % 8

                    if self.loading_time > 5:
                        self.current_msg = (self.current_msg + 1) % 4
                        self.loading_time = 0

                    image = self.draw.loading_screen(self.current_icon, self.current_msg)
            elif self.game_state == GameState.QR:
                image = self.draw.qr_screen(self.qr)

            self.fps_counter.step()
            self.draw_screen(image)

            if not self.listener.is_alive():
                self.video_writer.terminate()
                self.video_writer.join()
                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        help="Name of pose estimation model.",
        required=False,
        default="squat_wars/models/movenet_thunder_tpu.tflite",
    )

    parser.add_argument("--camera", help="Path to input video.", required=False, default=0)

    args = parser.parse_args()

    game = SquatGame(args.model, args.camera)
    game.run()
