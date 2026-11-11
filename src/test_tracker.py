
from tracker import SortTracker

tracker = SortTracker(
    max_age=5,
    min_hits=2
)

# Each entry represents one video frame.
frames = [
    [{"class": "car", "confidence": 0.95, "box": [100, 100, 60, 40]}],
    [{"class": "car", "confidence": 0.94, "box": [110, 100, 60, 40]}],
    [],  # Car temporarily disappears
    [],  # Still missing
    [{"class": "car", "confidence": 0.96, "box": [140, 100, 60, 40]}],
]

for frame_number, detections in enumerate(frames, start=1):

    tracks = tracker.update(detections)

    print(f"\nFrame {frame_number}")

    for track in tracks:

        print(
            f"ID: {track.track_id} | "
            f"Box: {track.box} | "
            f"Missed: {track.time_since_update}"
        )
