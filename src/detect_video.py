
from pathlib import Path
import time

import cv2

from detect_image import MODEL_PATH, detect_objects
from tracker import SortTracker


INPUT_VIDEO = Path("videos/drive.mp4")
OUTPUT_VIDEO = Path("runs/tracked.mp4")


def main():

    # 1. Check input video
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}"
        )

    # 2. Open video
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        raise RuntimeError("Could not open video")

    # 3. Read video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = video.get(cv2.CAP_PROP_FPS)

    if not 1 <= fps <= 120:
        fps = 20.0

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("Video width:", width)
    print("Video height:", height)
    print("Original FPS:", fps)
    print("Total frames:", total_frames)

    # 4. Prepare output directory
    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 5. Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        video.release()
        raise RuntimeError("Could not create output video")

    # 6. Load ONNX model
    print("Loading detection model...")

    network = cv2.dnn.readNetFromONNX(
        str(MODEL_PATH)
    )

    print("Model loaded!")

    # 7. Initialize tracker
    tracker = SortTracker(
        max_age=10,
        min_hits=3,
        max_cost=0.95
    )

    frame_count = 0
    start_time = time.perf_counter()

    window_name = "Mini Autopilot - Object Tracking"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    try:

        while True:

            # STEP 1: Read a video frame
            success, frame = video.read()

            if not success:
                print("End of video reached.")
                break

            # STEP 2: Detect objects
            detections = detect_objects(
                frame,
                network
            )

            # STEP 3: Update object tracker
            active_tracks = tracker.update(
                detections
            )

            # STEP 4: Draw bounding boxes
            for track in active_tracks:

                x, y, box_width, box_height = track.box

                if track.time_since_update == 0:
                    color = (0, 255, 0)
                    status = "DETECTED"
                else:
                    color = (0, 165, 255)
                    status = "PREDICTED"

                label = (
                    f"{track.class_name} "
                    f"ID:{track.track_id} "
                    f"{status}"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_width, y + box_height),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # STEP 5: Display tracking information
            cv2.putText(
                frame,
                f"Tracked objects: {len(active_tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # STEP 6: Save processed frame
            writer.write(frame)

            frame_count += 1

            # STEP 7: Calculate processing FPS
            if frame_count % 30 == 0:

                elapsed = time.perf_counter() - start_time

                processing_fps = frame_count / elapsed

                print(
                    f"Frames: {frame_count}/{total_frames} | "
                    f"Processing FPS: {processing_fps:.2f} | "
                    f"Visible objects: {len(active_tracks)}"
                )

            # STEP 8: DISPLAY EVERY FRAME
            cv2.imshow(window_name, frame)

            # Playback delay based on source video FPS
            delay = max(1, int(1000 / fps))

            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q") or key == 27:
                print("Stopped by user.")
                break

    finally:

        # STEP 9: Release resources
        video.release()
        writer.release()

        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print("\nTracking completed!")
    print("Frames processed:", frame_count)
    print("Time taken:", round(elapsed, 2), "seconds")

    if elapsed > 0:
        print(
            "Average processing FPS:",
            round(frame_count / elapsed, 2)
        )

    print("Output saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
