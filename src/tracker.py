
from dataclasses import dataclass
import math


@dataclass
class Track:
    track_id: int
    class_name: str
    box: list
    confidence: float
    missed_frames: int = 0
    hits: int = 1


class SimpleTracker:

    def __init__(self, max_missed=8):
        self.tracks = []
        self.next_id = 1
        self.max_missed = max_missed

    @staticmethod
    def calculate_iou(box_a, box_b):
        """
        Boxes use [x, y, width, height].
        """

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        # Find intersection coordinates
        left = max(ax, bx)
        top = max(ay, by)

        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)

        intersection_width = max(0, right - left)
        intersection_height = max(0, bottom - top)

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = aw * ah
        area_b = bw * bh

        union_area = area_a + area_b - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def calculate_center(box):
        x, y, width, height = box

        center_x = x + width / 2
        center_y = y + height / 2

        return center_x, center_y

    def update(self, detections, frame_shape):

        frame_height, frame_width = frame_shape[:2]

        # Allow some movement between consecutive frames
        max_distance = max(
            40,
            0.06 * math.hypot(frame_width, frame_height)
        )

        candidates = []

        # 1. Compare old tracks with new detections
        for track_index, track in enumerate(self.tracks):

            for detection_index, detection in enumerate(detections):

                # An object should not change its class
                if track.class_name != detection["class"]:
                    continue

                old_box = track.box
                new_box = detection["box"]

                # Calculate bounding box overlap
                iou = self.calculate_iou(old_box, new_box)

                # Calculate distance between centers
                old_center = self.calculate_center(old_box)
                new_center = self.calculate_center(new_box)

                distance = math.hypot(
                    old_center[0] - new_center[0],
                    old_center[1] - new_center[1]
                )

                # Reject unlikely matches
                if iou < 0.05 and distance > max_distance:
                    continue

                distance_score = max(
                    0,
                    1 - distance / max_distance
                )

                # Combine overlap and distance
                score = 0.7 * iou + 0.3 * distance_score

                candidates.append(
                    (score, track_index, detection_index)
                )

        # 2. Give the strongest matches priority
        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for score, track_index, detection_index in candidates:

            if track_index in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            track = self.tracks[track_index]
            detection = detections[detection_index]

            # Update the existing object's information
            track.box = detection["box"]
            track.confidence = detection["confidence"]
            track.missed_frames = 0
            track.hits += 1

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        # 3. Mark unmatched objects as temporarily missing
        for index, track in enumerate(self.tracks):

            if index not in matched_tracks:
                track.missed_frames += 1

        # 4. Remove objects missing for too long
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed
        ]

        # 5. Create new IDs for unmatched detections
        for index, detection in enumerate(detections):

            if index in matched_detections:
                continue

            new_track = Track(
                track_id=self.next_id,
                class_name=detection["class"],
                box=detection["box"],
                confidence=detection["confidence"]
            )

            self.tracks.append(new_track)
            self.next_id += 1

        # Return currently visible objects
        return [
            track
            for track in self.tracks
            if track.missed_frames == 0
        ]
