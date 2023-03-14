from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import utils

from squat_wars.classifier import Classifier
from squat_wars.data import BodyPart, Person
from squat_wars.game_state import WINDOW_HEIGHT, WINDOW_WIDTH
from squat_wars.metrics import FPS
from squat_wars.player import Player
from squat_wars.ranking import Ranking


class DrawEngine:
    def __init__(self, classifier: Classifier, ranking: Ranking, fps_counter: FPS) -> None:
        self.classifier = classifier
        self.ranking = ranking
        self.fps_counter = fps_counter
        self.prev_squat_count = classifier.squat_count
        self.cooldown = 0
        self.load_images()

    def load_images(self):
        """Loads the different images needed for the game to work"""
        images = Path("squat_wars", "assets", "images")

        self.bg = cv2.imread(str(images / "base.png"))
        self.squat = cv2.imread(str(images / "squat.png"))
        self.loading = cv2.imread(str(images / "names.png"), flags=cv2.IMREAD_UNCHANGED)
        self.scan = cv2.imread(str(images / "scan.png"))[:WINDOW_HEIGHT, :WINDOW_WIDTH]
        self.end_bg = cv2.imread(str(images / "0seconds.png"))
        self.final = cv2.imread(str(images / "ranking.png"))

        # load phrase images
        self.new_loading = [
            cv2.imread(str(images / "loading" / f"{i}.png"))[:WINDOW_HEIGHT, :WINDOW_WIDTH]
            for i in range(4)
        ]

        # load spinning wheel
        self.loading_icon = [
            cv2.resize(
                cv2.imread(str(images / "icon" / f"{i}.png"), flags=cv2.IMREAD_UNCHANGED), (96, 96)
            )
            for i in range(1, 9)
        ]

        # load countdown images
        self.count = {
            i: cv2.imread(str(images / "countdown" / f"{i}.png"), flags=cv2.IMREAD_UNCHANGED)
            for i in range(1, 4)
        }

    def alpha_blend(self, rgb: np.ndarray, rgba: np.ndarray):
        rgb_float = rgb.astype(float)
        rgba_float = rgba.astype(float)
        alpha = rgba[:, :, 3][..., np.newaxis] / 255
        return rgb_float * (1 - alpha) + rgba_float[:, :, :3] * alpha

    def draw_fps(self, image: np.ndarray):
        cv2.putText(
            img=image,
            text=f"FPS: {self.fps_counter.fps:.2f}",
            org=(10, 20),
            fontFace=cv2.FONT_ITALIC,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    def draw_ranking(self, image: np.ndarray) -> None:
        for i, rank_spot in enumerate(self.ranking[:6]):
            (text_w, text_h), _ = cv2.getTextSize(
                text=rank_spot.player.name,
                fontFace=cv2.FONT_ITALIC,
                fontScale=1,
                thickness=1,
            )
            (score_w, score_h), _ = cv2.getTextSize(
                text=str(rank_spot.score),
                fontFace=cv2.FONT_ITALIC,
                fontScale=1,
                thickness=1,
            )
            # 32 is the distance in pixels between the lines
            y_pos = 1640 + 32 * i

            # 70 is the width of the score line
            x_off = (70 - score_w) // 2

            # TODO recheck
            player_name = (
                rank_spot.player.name
                if len(rank_spot.player.name) <= 15
                else rank_spot.player.name[:12] + "..."
            )

            cv2.putText(
                img=image,
                text=player_name,
                org=(396, y_pos),
                fontFace=cv2.FONT_ITALIC,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img=image,
                text=str(rank_spot.score),
                org=(673 + x_off, y_pos),
                fontFace=cv2.FONT_ITALIC,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def draw_camera_in_background(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        self.draw_fps(frame)
        height, width, _ = frame.shape
        bg_height, bg_width, _ = background.shape
        height_offset = 643
        width_offset = int((bg_width - width) / 2)

        background[
            height_offset : height_offset + height, width_offset : width_offset + width, :
        ] = frame[:, :, :]

        return background

    def draw_image_from_center(
        self, image: np.ndarray, background: np.ndarray, x_off: int = 0, y_off: int = 0
    ) -> np.ndarray:
        im_height, im_width, im_ch = image.shape
        bg_height, bg_width, _ = background.shape
        h_offset = (bg_height - im_height) // 2 + y_off
        w_offset = (bg_width - im_width) // 2 + x_off

        section = background[h_offset : h_offset + im_height, w_offset : w_offset + im_width, :]
        section[:, :, :] = self.alpha_blend(section[:, :, :], image) if im_ch == 4 else image

        return background

    def draw_overlay(self, background: np.ndarray, alpha: float = 0.8):
        bg_height, bg_width, _ = background.shape
        overlay_color = np.array([[[35, 7, 11]]], dtype=np.uint8)  # hex #0B0723 in bgr
        overlay = np.broadcast_to(overlay_color, (bg_height, bg_width, 3))
        return cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)

    def distance_lines(self, image: np.ndarray, person: Person) -> np.ndarray:
        """
        Draws the distance lines on the image.
        One for squat distance and another for normal distance.

        Parameters
        ----------
        image : np.ndarray
            The image to draw on.
        person : Person
            The person to draw the lines for.

        Returns
        -------
        np.ndarray
            The image with the lines drawn on it.
        """
        left_knee = person.keypoints[BodyPart.LEFT_KNEE.value]

        if self.classifier.squat_distance_pixel is None:
            return image

        bbox_start_x = person.bounding_box.start_point.x
        bbox_end_x = person.bounding_box.end_point.x

        squat_line = left_knee.coordinate.y - self.classifier.squat_distance_pixel
        cv2.line(image, (bbox_start_x, squat_line), (bbox_end_x, squat_line), (255, 255, 0), 2)

        normal_line = left_knee.coordinate.y - self.classifier.normal_distance_pixel
        cv2.line(image, (bbox_start_x, normal_line), (bbox_end_x, normal_line), (0, 0, 255), 2)

        return image

    def draw_person(self, image: np.ndarray, person: Optional[Person] = None) -> np.ndarray:
        if person:
            image = utils.visualize(image, [person])
            if self.classifier.max_vertical_distance > 0:
                image = self.distance_lines(image, person)
        return image

    def not_started_screen(self, frame: np.ndarray) -> np.ndarray:
        bg = self.bg.copy()
        self.draw_ranking(bg)
        self.draw_camera_in_background(frame, bg)
        return bg

    def user_selection_screen(
        self, frame: np.ndarray, player_name: str, player_email: str
    ) -> np.ndarray:
        bg = self.bg.copy()
        self.draw_ranking(bg)
        self.draw_camera_in_background(frame, bg)

        # render only last 20 characters
        name = player_name[-20:]
        email = player_email[-20:]

        bg_height, bg_width, _ = bg.shape
        box_top = (bg_height - self.loading.shape[0]) // 2

        # draw overlay
        bg = self.draw_overlay(bg)

        # draw loading player box
        self.draw_image_from_center(self.loading, bg)

        # draw centered player name
        (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_ITALIC, 1, 2)
        name_x_off = (bg_width - text_width) // 2
        name_y_off = box_top + 152 + text_height // 2
        cv2.putText(
            img=bg,
            text=name,
            org=(name_x_off, name_y_off),
            fontFace=cv2.FONT_ITALIC,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # draw centered player email
        (email_width, email_height), _ = cv2.getTextSize(email, cv2.FONT_ITALIC, 1, 2)
        email_x_off = (bg_width - email_width) // 2
        email_y_off = box_top + 272 + email_height // 2
        cv2.putText(
            img=bg,
            text=email,
            org=(email_x_off, email_y_off),
            fontFace=cv2.FONT_ITALIC,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return bg

    def countdown(self, frame: np.ndarray, remaining_time: int) -> np.ndarray:
        bg = self.bg.copy()
        self.draw_ranking(bg)
        self.draw_camera_in_background(frame, bg)

        # draw overlay
        bg = self.draw_overlay(bg)

        self.draw_image_from_center(self.count[remaining_time], bg)

        return bg

    def started_screen(self, frame: np.ndarray, player: Player, time_left: int) -> np.ndarray:
        # blink for 3 frames if a squat was made
        if self.prev_squat_count < self.classifier.squat_count:
            self.cooldown = 3
        self.prev_squat_count = self.classifier.squat_count

        if self.cooldown > 0:
            bg = self.squat
            self.cooldown -= 1
        elif time_left > 0:
            bg = self.bg
        else:
            bg = self.end_bg

        bg = bg.copy()
        self.draw_ranking(bg)
        self.draw_camera_in_background(frame, bg)

        config = {
            "fontFace": cv2.FONT_ITALIC,
            "fontScale": 1,
            "color": (255, 255, 255),
            "thickness": 2,
            "lineType": cv2.LINE_AA,
        }

        # draw player name
        (text_width, text_height), _ = cv2.getTextSize(player.name, cv2.FONT_ITALIC, 1, 2)
        cv2.putText(
            img=bg,
            text=player.name,
            org=((bg.shape[1] - text_width) // 2, 429 + text_height // 2),
            **config,
        )

        # draw time left
        (time_width, time_height), _ = cv2.getTextSize(str(time_left), cv2.FONT_ITALIC, 1, 2)
        cv2.putText(
            img=bg,
            text=str(time_left),
            org=(418 - time_width // 2, 553 + time_height // 2),
            **config,
        )

        # draw squat count
        (count_width, count_height), _ = cv2.getTextSize(
            str(self.classifier.squat_count), cv2.FONT_ITALIC, 1, 2
        )
        cv2.putText(
            img=bg,
            text=str(self.classifier.squat_count),
            org=(660 - count_width // 2, 553 + count_height // 2),
            **config,
        )

        return bg

    def loading_screen(self, icon_index: int, msg_index: int):
        bg = self.new_loading[msg_index].copy()
        icon = self.loading_icon[icon_index]
        self.draw_image_from_center(icon, bg, y_off=60)

        return bg

    def qr_screen(self, qr: np.ndarray):
        bg = self.scan.copy()
        return self.draw_image_from_center(qr, bg, x_off=-11, y_off=83)

    def ending_screen(self, player: Player):
        bg = self.final.copy()
        ranking_pos = self.ranking.get_player_position(player)

        config = {
            "fontFace": cv2.FONT_ITALIC,
            "fontScale": 1,
            "color": (255, 255, 255),
            "thickness": 2,
            "lineType": cv2.LINE_AA,
        }

        # draw player name
        (text_width, text_height), _ = cv2.getTextSize(player.name, cv2.FONT_ITALIC, 1, 2)
        cv2.putText(
            img=bg,
            text=player.name,
            org=(548 - text_width // 2, 960 + text_height // 2),
            **config,
        )

        # draw time left
        (time_width, time_height), _ = cv2.getTextSize(str(ranking_pos), cv2.FONT_ITALIC, 1, 2)
        cv2.putText(
            img=bg,
            text=str(ranking_pos),
            org=(423 - time_width // 2, 1095 + time_height // 2),
            **config,
        )

        # draw squat count
        (count_width, count_height), _ = cv2.getTextSize(
            str(self.classifier.squat_count), cv2.FONT_ITALIC, 1, 2
        )
        cv2.putText(
            img=bg,
            text=str(self.classifier.squat_count),
            org=(676 - count_width // 2, 1095 + count_height // 2),
            **config,
        )

        return bg
