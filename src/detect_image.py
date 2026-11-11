
from pathlib import Path

import cv2
import numpy as np


# Model configuration
MODEL_PATH = Path("models/yolov8n.onnx")
IMAGE_PATH = Path("videos/frame.jpg")

INPUT_WIDTH = 640
INPUT_HEIGHT = 480

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


# COCO dataset class labels
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def letterbox(image):
    """Resize the image without changing its aspect ratio."""

    original_height, original_width = image.shape[:2]

    scale = min(
        INPUT_WIDTH / original_width,
        INPUT_HEIGHT / original_height
    )

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    resized = cv2.resize(
        image,
        (resized_width, resized_height)
    )

    padding_width = INPUT_WIDTH - resized_width
    padding_height = INPUT_HEIGHT - resized_height

    left = padding_width // 2
    right = padding_width - left

    top = padding_height // 2
    bottom = padding_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return padded, scale, left, top


def detect_objects(image, network):
    """Run YOLO inference and decode its output."""

    original_height, original_width = image.shape[:2]

    # Prepare image for the neural network
    processed, scale, pad_left, pad_top = letterbox(image)

    blob = cv2.dnn.blobFromImage(
        processed,
        scalefactor=1 / 255.0,
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False
    )

    # Neural network inference
    network.setInput(blob)
    output = network.forward()

    print("Raw output shape:", output.shape)

    # YOLOv8 output: (1, 84, 6300)
    # Transpose to: (6300, 84)
    predictions = output[0].T

    boxes = []
    confidences = []
    class_ids = []

    for prediction in predictions:

        # First 4 values: center_x, center_y, width, height
        # Remaining 80 values: class scores
        scores = prediction[4:]

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        center_x, center_y, box_width, box_height = prediction[:4]

        # Convert center coordinates into corner coordinates
        x1 = (center_x - box_width / 2 - pad_left) / scale
        y1 = (center_y - box_height / 2 - pad_top) / scale

        x2 = (center_x + box_width / 2 - pad_left) / scale
        y2 = (center_y + box_height / 2 - pad_top) / scale

        # Keep the box inside the original image
        x1 = int(np.clip(x1, 0, original_width - 1))
        y1 = int(np.clip(y1, 0, original_height - 1))
        x2 = int(np.clip(x2, 0, original_width - 1))
        y2 = int(np.clip(y2, 0, original_height - 1))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        boxes.append([x1, y1, box_w, box_h])
        confidences.append(confidence)
        class_ids.append(class_id)

    # Remove duplicate, overlapping detections
    selected = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []

    if len(selected) > 0:
        for index in np.array(selected).flatten():

            x, y, width, height = boxes[index]

            detections.append({
                "class": CLASSES[class_ids[index]],
                "confidence": confidences[index],
                "box": [x, y, width, height]
            })

    return detections


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Could not read input image")

    print("Loading ONNX model...")

    network = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

    print("Model loaded successfully!")

    detections = detect_objects(image, network)

    for item in detections:
        x, y, width, height = item["box"]

        label = (
            f'{item["class"]} '
            f'{item["confidence"]:.2f}'
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        print(item)

    output_path = Path("runs/detection.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)

    print(f"\nOutput saved to: {output_path}")
    print(f"Total detections: {len(detections)}")

    cv2.imshow("Mini Autopilot - Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
