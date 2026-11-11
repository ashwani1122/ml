
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------
# 1. Bounding-box utilities
# --------------------------------------------------

def box_to_measurement(box):
    """Convert [x, y, w, h] to [cx, cy, w, h]."""

    x, y, width, height = box

    center_x = x + width / 2
    center_y = y + height / 2

    return np.array(
        [[center_x], [center_y], [width], [height]],
        dtype=np.float32
    )


def state_to_box(state):
    """Convert Kalman state to [x, y, w, h]."""

    center_x, center_y, width, height = state[:4].flatten()

    width = max(2.0, float(width))
    height = max(2.0, float(height))

    x = float(center_x) - width / 2
    y = float(center_y) - height / 2

    return [x, y, width, height]


def calculate_iou(box_a, box_b):
    """Intersection over Union for [x, y, w, h] boxes."""

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
    """Distance between bounding-box centers."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    center_a = (ax + aw / 2, ay + ah / 2)
    center_b = (bx + bw / 2, by + bh / 2)

    return float(np.hypot(
        center_a[0] - center_b[0],
        center_a[1] - center_b[1]
    ))


# --------------------------------------------------
# 2. Appearance feature extraction
# --------------------------------------------------

def extract_appearance(frame, box):
    """
    Extract an HSV color histogram from a detection.

    This is a lightweight appearance descriptor.
    It is not a learned neural-network embedding.
    """

    image_height, image_width = frame.shape[:2]

    x, y, width, height = box

    x1 = max(0, int(x))
    y1 = max(0, int(y))

    x2 = min(image_width, int(x + width))
    y2 = min(image_height, int(y + height))

    # Reject invalid boxes
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    # Use the central region to reduce background influence.
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)

    x1 += margin_x
    x2 -= margin_x
    y1 += margin_y
    y2 -= margin_y

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Convert BGR to HSV.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Create a two-dimensional hue/saturation histogram.
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256]
    )

    # Normalize histogram.
    cv2.normalize(
        histogram,
        histogram,
        alpha=1.0,
        norm_type=cv2.NORM_L1
    )

    return histogram.astype(np.float32)


def appearance_distance(hist_a, hist_b):
    """
    Compare two appearance histograms.

    Bhattacharyya distance:
    0 = very similar
    1 = very different
    """

    if hist_a is None or hist_b is None:
        return None

    return float(
        cv2.compareHist(
            hist_a,
            hist_b,
            cv2.HISTCMP_BHATTACHARYYA
        )
    )


# --------------------------------------------------
# 3. Kalman Filter
# --------------------------------------------------

def create_kalman_filter(box):

    # 8 state variables, 4 observed variables.
    kalman = cv2.KalmanFilter(8, 4)

    # [cx, cy, w, h, vx, vy, vw, vh]
    transition = np.eye(8, dtype=np.float32)

    transition[0, 4] = 1.0
    transition[1, 5] = 1.0
    transition[2, 6] = 1.0
    transition[3, 7] = 1.0

    kalman.transitionMatrix = transition

    measurement = np.zeros((4, 8), dtype=np.float32)
    measurement[:4, :4] = np.eye(4, dtype=np.float32)

    kalman.measurementMatrix = measurement

    kalman.processNoiseCov = np.diag(
        [1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]
    ).astype(np.float32)

    kalman.measurementNoiseCov = np.diag(
        [10, 10, 20, 20]
    ).astype(np.float32)

    kalman.errorCovPost = np.diag(
        [10, 10, 10, 10, 100, 100, 100, 100]
    ).astype(np.float32)

    kalman.statePost = np.vstack([
        box_to_measurement(box),
        np.zeros((4, 1), dtype=np.float32)
    ]).astype(np.float32)

    return kalman


# --------------------------------------------------
# 4. Track object
# --------------------------------------------------

@dataclass
class Track:

    track_id: int
    class_name: str
    kalman: object
    confidence: float

    appearance: np.ndarray = None

    age: int = 1
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    confirmed: bool = False

    predicted_box: list = None

    @property
    def box(self):

        values = state_to_box(self.kalman.statePost)

        return [
            int(round(values[0])),
            int(round(values[1])),
            int(round(values[2])),
            int(round(values[3]))
        ]

    def predict(self):

        prediction = self.kalman.predict()

        self.age += 1
        self.time_since_update += 1

        self.predicted_box = state_to_box(prediction)

        return self.predicted_box

    def correct(self, detection, new_appearance, min_hits):

        measurement = box_to_measurement(
            detection["box"]
        )

        self.kalman.correct(measurement)

        self.confidence = detection["confidence"]

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Update appearance using an exponential moving average.
        if new_appearance is not None:

            if self.appearance is None:
                self.appearance = new_appearance.copy()

            else:
                self.appearance = (
                    0.9 * self.appearance
                    + 0.1 * new_appearance
                )

                cv2.normalize(
                    self.appearance,
                    self.appearance,
                    alpha=1.0,
                    norm_type=cv2.NORM_L1
                )

        if self.hit_streak >= min_hits:
            self.confirmed = True

    def mark_missed(self):

        # Use the Kalman prediction when detection is absent.
        self.kalman.statePost = self.kalman.statePre.copy()

        self.kalman.errorCovPost = (
            self.kalman.errorCovPre.copy()
        )

        self.hit_streak = 0


# --------------------------------------------------
# 5. Appearance-aware SORT tracker
# --------------------------------------------------

class SortTracker:

    INVALID_COST = 1_000_000.0

    def __init__(
        self,
        max_age=10,
        min_hits=3,
        max_cost=0.85
    ):

        self.tracks = []

        self.next_id = 1

        self.max_age = max_age
        self.min_hits = min_hits
        self.max_cost = max_cost

    def create_track(self, detection, appearance):

        track = Track(
            track_id=self.next_id,
            class_name=detection["class"],
            kalman=create_kalman_filter(
                detection["box"]
            ),
            confidence=detection["confidence"],
            appearance=appearance
        )

        if self.min_hits <= 1:
            track.confirmed = True

        self.next_id += 1

        return track

    def build_cost_matrix(self, detections, features):

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

                # Match only objects of the same class.
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

                appearance = appearance_distance(
                    track.appearance,
                    features[j]
                )

                # Reject implausible matches.
                if distance > 2.0 * distance_limit:
                    continue

                if (
                    iou < 0.01
                    and distance > distance_limit
                    and (
                        appearance is None
                        or appearance > 0.25
                    )
                ):
                    continue

                normalized_distance = min(
                    distance / distance_limit,
                    1.0
                )

                # Cost is lower when objects look and move similarly.
                if appearance is None:

                    cost = (
                        0.55 * (1.0 - iou)
                        + 0.45 * normalized_distance
                    )

                else:

                    cost = (
                        0.40 * (1.0 - iou)
                        + 0.25 * normalized_distance
                        + 0.35 * appearance
                    )

                cost_matrix[i, j] = cost

        return cost_matrix

    def update(self, detections, frame=None):

        # STEP 1: Extract appearance descriptors.
        if frame is not None:

            features = [
                extract_appearance(frame, detection["box"])
                for detection in detections
            ]

        else:
            features = [None] * len(detections)

        # STEP 2: Predict existing tracks.
        for track in self.tracks:
            track.predict()

        matched_tracks = set()
        matched_detections = set()

        # STEP 3: Build cost matrix and globally assign matches.
        if self.tracks and detections:

            cost_matrix = self.build_cost_matrix(
                detections,
                features
            )

            track_indices, detection_indices = (
                linear_sum_assignment(cost_matrix)
            )

            # STEP 4: Update matched tracks.
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

                track.correct(
                    detections[detection_index],
                    features[detection_index],
                    self.min_hits
                )

                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

        # STEP 5: Preserve predictions for unmatched tracks.
        for index, track in enumerate(self.tracks):

            if index not in matched_tracks:
                track.mark_missed()

        # STEP 6: Remove expired tracks.
        self.tracks = [
            track
            for track in self.tracks
            if track.time_since_update <= self.max_age
        ]

        # STEP 7: Create tracks for unmatched detections.
        for index, detection in enumerate(detections):

            if index not in matched_detections:

                self.tracks.append(
                    self.create_track(
                        detection,
                        features[index]
                    )
                )

        # STEP 8: Return confirmed tracks.
        return [
            track
            for track in self.tracks
            if track.confirmed
            and track.time_since_update <= self.max_age
        ]
