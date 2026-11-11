
import cv2
import numpy as np

# Load our saved camera image
image = cv2.imread("videos/frame.jpg")

if image is None:
    raise FileNotFoundError("Could not load videos/frame.jpg")

# Print image properties
print("Image shape:", image.shape)
print("Data type:", image.dtype)

# Access an individual pixel
pixel = image[120, 120]
print("Pixel at (120, 120):", pixel)

# Calculate average pixel intensity
average = np.mean(image)
print("Average pixel intensity:", average)

# Convert BGR image into RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("RGB pixel:", rgb_image[100, 100])
