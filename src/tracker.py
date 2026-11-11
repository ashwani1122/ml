
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_to_measurement(box):
    """
    Convert [x, y, width, height] into:

    [center_x, center_y, width, height]
    """

    x, y, width, height = box

    center_x = x + width / 2
    center_y = y + height / 2

    return np.array(
        [[center_x], [center_y], [width], [height]],
        dtype=np.float32
    )


def state_to_box(state):
    """
    Convert Kalman state into [x, y, width, height].
    """

    center_x, center_y, width, height = state[:4].flatten()

    width = max(2.0, float(width))
    height = max(2.0, float(height))

    x = float(center_x) - width / 2
    y = float(center_y) - height / 2

    return [x, y, width, height]


def create_kalman_filter(box):
    """
    State:
    [cx, cy, width, height, vx, vy, vwidth, vheight]

    Measurement:
    [cx, cy, width, height]
    """

    kalman = cv2.KalmanFilter(8, 4)

    # State transition matrix.
    # Position at the next frame = current position + velocity.
    transition = np.eye(8, dtype=np.float32)

    transition[0, 4] = 1.0
    transition[1, 5] = 1.0
    transition[2, 6] = 1.0
    transition[3, 7] = 1.0

    kalman.transitionMatrix = transition

    # We observe only position and box dimensions.
    measurement = np.zeros((4, 8), dtype=np.float32)
    measurement[:4, :4] = np.eye(4, dtype=np.float32)

    kalman.measurementMatrix = measurement

    # Process noise: uncertainty in the motion model.
    kalman.processNoiseCov = np.diag(
        [1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]
    ).astype(np.float32)

    # Measurement noise: uncertainty in detector output.
    kalman.measurementNoiseCov = np.diag(
        [10, 10, 20, 20]
    ).astype(np.float32)

    # Initial uncertainty.
    kalman.errorCovPost = np.diag(
        [10, 10, 10, 10, 100, 100, 100, 100]
    ).astype(np.float32)

    # Initial position comes from the first detection.
    kalman.statePost = np.vstack(
        [
            box_to_measurement(box),
            np.zeros((4, 1), dtype=np.float32)
        ]
    ).astype(np.float32)

    return kalman


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union.
    Both boxes use [x, y, width, height].
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)

    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)

    intersection = intersection_width * intersection_height

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def center_distance(box_a, box_b):
    """Measure distance between two box centers."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    center_a_x = ax + aw / 2
    center_a_y = ay + ah / 2

    center_b_x = bx + bw / 2
    center_b_y = by + bh / 2

    return float(np.hypot(
        center_a_x - center_b_x,
        center_a_y - center_b_y
    ))


@dataclass
class Track:

    track_id: int
    class_name: str
    kalman: object
    confidence: float

    age: int = 1
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    confirmed: bool = False

    predicted_box: list = None

    @property
    def box(self):
        """Get the latest corrected or predicted box."""

        values = state_to_box(self.kalman.statePost)

        return [
            int(round(values[0])),
            int(round(values[1])),
            int(round(values[2])),
            int(round(values[3]))
        ]

    def predict(self):
        """Predict the next position using the Kalman filter."""

        prediction = self.kalman.predict()

        self.age += 1
        self.time_since_update += 1

        self.predicted_box = state_to_box(prediction)

        return self.predicted_box

    def correct(self, detection, min_hits):
        """Correct the predicted position with a new detection."""

        measurement = box_to_measurement(detection["box"])

        self.kalman.correct(measurement)

        self.confidence = detection["confidence"]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.hit_streak >= min_hits:
            self.confirmed = True

    def mark_missed(self):
        """
        Keep the prediction when the detector misses the object.

        OpenCV's next prediction must start from the previous
        predicted state when no correction is available.
        """

        self.kalman.statePost = self.kalman.statePre.copy()
        self.kalman.errorCovPost = self.kalman.errorCovPre.copy()

        self.hit_streak = 0


class SortTracker:

    INVALID_COST = 1_000_000.0

    def __init__(
        self,
        max_age=10,
        min_hits=3,
        max_cost=0.95
    ):

        self.tracks = []

        self.next_id = 1

        self.max_age = max_age
        self.min_hits = min_hits
        self.max_cost = max_cost

    def create_track(self, detection):

        track = Track(
            track_id=self.next_id,
            class_name=detection["class"],
            kalman=create_kalman_filter(detection["box"]),
            confidence=detection["confidence"]
        )

        if self.min_hits <= 1:
            track.confirmed = True

        self.next_id += 1

        return track

    def build_cost_matrix(self, detections):

        rows = len(self.tracks)
        columns = len(detections)

        cost_matrix = np.full(
            (rows, columns),
            self.INVALID_COST,
            dtype=np.float32
        )

        for i, track in enumerate(self.tracks):

            predicted_box = track.predicted_box

            for j, detection in enumerate(detections):

                # A car should not match a pedestrian.
                if track.class_name != detection["class"]:
                    continue

                detected_box = detection["box"]

                iou = calculate_iou(
                    predicted_box,
                    detected_box
                )

                distance = center_distance(
                    predicted_box,
                    detected_box
                )

                # Dynamic distance limit based on box sizes.
                predicted_size = np.hypot(
                    predicted_box[2],
                    predicted_box[3]
                )

                detected_size = np.hypot(
                    detected_box[2],
                    detected_box[3]
                )

                distance_limit = max(
                    50.0,
                    0.5 * (predicted_size + detected_size)
                )

                # Reject detections that are too far away
                # and have no meaningful box overlap.
                if iou < 0.01 and distance > distance_limit:
                    continue

                normalized_distance = min(
                    distance / distance_limit,
                    1.0
                )

                # Lower cost means a better match.
                cost = (
                    0.7 * (1.0 - iou)
                    + 0.3 * normalized_distance
                )

                cost_matrix[i, j] = cost

        return cost_matrix

    def update(self, detections):

        # STEP 1: Predict every existing track.
        for track in self.tracks:
            track.predict()

        matched_tracks = set()
        matched_detections = set()

        # STEP 2: Create the assignment cost matrix.
        if self.tracks and detections:

            cost_matrix = self.build_cost_matrix(detections)

            # STEP 3: Find globally optimal assignments.
            track_indices, detection_indices = (
                linear_sum_assignment(cost_matrix)
            )

            # STEP 4: Correct valid matches.
            for track_index, detection_index in zip(
                track_indices,
                detection_indices
            ):

                cost = cost_matrix[
                    track_index,
                    detection_index
                ]

                if cost >= self.max_cost:
                    continue

                track = self.tracks[track_index]

                detection = detections[detection_index]

                track.correct(
                    detection,
                    self.min_hits
                )

                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

        # STEP 5: Keep predictions for unmatched tracks.
        for index, track in enumerate(self.tracks):

            if index not in matched_tracks:
                track.mark_missed()

        # STEP 6: Remove tracks missing for too long.
        self.tracks = [
            track
            for track in self.tracks
            if track.time_since_update <= self.max_age
        ]

        # STEP 7: Create tracks for unmatched detections.
        for index, detection in enumerate(detections):

            if index not in matched_detections:
                self.tracks.append(
                    self.create_track(detection)
                )

        # STEP 8: Return confirmed tracks.
        # We include predictions during temporary occlusions.
        return [
            track
            for track in self.tracks
            if track.confirmed
            and track.time_since_update <= self.max_age
        ]
