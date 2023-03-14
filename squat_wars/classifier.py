from enum import Enum
from typing import Optional

from squat_wars.data import BodyPart, Person


class Pose(Enum):
    STANDING = 0
    TRANSITION = 1
    SQUAT = 2


class Classifier:
    """Classifies a person based on their pose."""

    # create init from base classifier
    def __init__(self, inertia: int = 3) -> None:
        self.inertia = inertia

        # Vertical distance between hip and knee to classify as standing position.
        # Unit is normalized. 1 unit corresponds to 100% of the height of the bounding box
        self.normal_distance = 0.24
        # Vertical distance between hip and knee to classify as squat position
        self.squat_distance = 0.17

        # set initial state
        self.reset()

    @property
    def max_vertical_distance(self) -> float:
        return max(self.vertical_distances)

    def update_distance(self, distance: float):
        self.vertical_distances[self.current_frame] = distance
        self.current_frame = (self.current_frame + 1) % 45

    @property
    def normal_distance_pixel(self) -> int:
        return round(self.normal_distance * self.max_vertical_distance)

    @property
    def squat_distance_pixel(self) -> int:
        return round(self.squat_distance * self.max_vertical_distance)

    def reset(self):
        """Resets the classifier."""
        self.current_frame = 0
        self.vertical_distances = [0.0] * 45  # keep heights of last 45 frames

        self.frame_counter = 0  # Number of frames in a row with the same classification
        self.previous_classification = None
        self.current_classification = None
        self.previous_pose_inertia = None  # Previous pose with inertia
        self.current_pose_inertia = None  # Current pose with inertia
        self.current_pose_with_transition_inertia = None  # Current pose with inertia and transition
        self.squat_count = 0  # Number of squats performed

    def classify_pose(self, person: Person) -> Pose:
        """Classifies a person based on their pose."""
        keypoints = person.keypoints

        left_hip_y = keypoints[BodyPart.LEFT_HIP.value].coordinate.y
        right_hip_y = keypoints[BodyPart.RIGHT_HIP.value].coordinate.y

        left_knee_y = keypoints[BodyPart.LEFT_KNEE.value].coordinate.y
        right_knee_y = keypoints[BodyPart.RIGHT_KNEE.value].coordinate.y

        bbox = person.bounding_box
        self.update_distance(abs(bbox.end_point.y - bbox.start_point.y))

        left_distance = abs(left_knee_y - left_hip_y) / self.max_vertical_distance
        right_distance = abs(right_knee_y - right_hip_y) / self.max_vertical_distance

        if min(left_distance, right_distance) >= self.normal_distance:
            return Pose.STANDING

        if max(left_distance, right_distance) <= self.squat_distance:
            return Pose.SQUAT

        return Pose.TRANSITION

    def classify(self, person: Person, count: bool = False) -> Optional[Pose]:
        """Classifies a person based on their pose and optionally counts squats."""

        # check if there are more than 4 keypoints with low confidence.
        if sum([keypoint.score < 0.25 for keypoint in person.keypoints]) > 4:
            self.current_pose_with_transition_inertia = None
            return None

        classification = self.classify_pose(person)

        if self.previous_classification is None:
            self.previous_classification = classification

        if classification == self.previous_classification:
            self.frame_counter += 1
        else:
            self.frame_counter = 0

        self.previous_classification = classification

        if self.frame_counter >= self.inertia:
            self.previous_pose_inertia = self.current_pose_inertia
            if classification != Pose.TRANSITION:
                self.current_pose_inertia = classification
            self.current_pose_with_transition_inertia = classification

        if count:
            if (
                self.current_pose_inertia == Pose.SQUAT
                and self.previous_pose_inertia == Pose.STANDING
            ):
                self.squat_count += 1

        return self.current_pose_with_transition_inertia
