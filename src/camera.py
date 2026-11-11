
import cv2
from pathlib import Path

# 1. Create the videos directory if it doesn't exist
video_directory = Path("videos")
video_directory.mkdir(parents=True, exist_ok=True)

output_path = video_directory / "drive.mp4"

# 2. Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened")

print("Camera started!")

# 3. Recording variables
video_writer = None
recording = False

try:
    while True:

        # 4. Capture the next frame
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame")
            break

        height, width = frame.shape[:2]

        # 5. Get keyboard input
        key = cv2.waitKey(1) & 0xFF

        # 6. Start recording when R is pressed
        if key == ord("r") and not recording:

            # Get camera FPS, with a fallback
            fps = camera.get(cv2.CAP_PROP_FPS)

            if fps <= 0 or fps > 120:
                fps = 20.0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            if not video_writer.isOpened():
                raise RuntimeError("Could not initialize video writer")

            recording = True
            print("Recording started!")

        # 7. Write frames into the video
        if recording:
            video_writer.write(frame)

            cv2.putText(
                frame,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # 8. Stop recording when S is pressed
        if key == ord("s") and recording:

            video_writer.release()
            video_writer = None

            recording = False

            print("Recording stopped!")
            print(f"Video saved at: {output_path}")

        # 9. Display camera window
        cv2.imshow("Mini Autopilot Camera", frame)

        # 10. Exit when Q or ESC is pressed
        if key == ord("q") or key == 27:
            print("Exiting...")
            break

finally:

    # 11. Release resources
    if video_writer is not None:
        video_writer.release()

    camera.release()
    cv2.destroyAllWindows()
