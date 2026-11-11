
from pathlib import Path
import time

import cv2

from detect_image import MODEL_PATH, detect_objects
from tracker import SimpleTracker


INPUT_VIDEO = Path("videos/drive.mp4")
OUTPUT_VIDEO = Path("runs/tracked.mp4")


def main():

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}"
        )

    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        raise RuntimeError("Could not open video")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = video.get(cv2.CAP_PROP_FPS)

    if not 1 <= fps <= 120:
        fps = 20.0

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    # Load model only once
    network = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

    # Create our tracker
    tracker = SimpleTracker(max_missed=8)

    frame_count = 0
    start_time = time.perf_counter()

    try:

        while True:

            success, frame = video.read()

            if not success:
                break

            # 1. Detect objects in this frame
            detections = detect_objects(frame, network)

            # 2. Match detections with existing tracks
            active_tracks = tracker.update(
                detections,
                frame.shape
            )

            # 3. Draw track information
            for track in active_tracks:

                x, y, box_width, box_height = track.box

                label = (
                    f"{track.class_name} "
                    f"ID:{track.track_id} "
                    f"{track.confidence:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_width, y + box_height),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # 4. Show number of currently visible tracks
            cv2.putText(
                frame,
                f"Tracked objects: {len(active_tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # 5. Save processed frame
            writer.write(frame)

            frame_count += 1

            if frame_count % 30 == 0:

                elapsed = time.perf_counter() - start_time

                processing_fps = frame_count / elapsed

                print(
                    f"Frames: {frame_count} | "
                    f"Tracking FPS: {processing_fps:.2f} | "
                    f"Visible objects: {len(active_tracks)}"
                )

            cv2.imshow("Mini Autopilot - Object Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\nTracking completed!")
    print("Frames processed:", frame_count)
    print("Output saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
