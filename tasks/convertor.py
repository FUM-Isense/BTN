import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('dishup.png', cv2.IMREAD_GRAYSCALE)

# Convert the grayscale image to a binary image
# Thresholding: set white pixels (255) to 0, and black pixels (0) to 1
_, binary_image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)

# Convert the binary image (0s and 1s) back to 0s and 255s to display properly
display_image = binary_image * 255

resized_image = cv2.resize(display_image, (200, 57), interpolation=cv2.INTER_NEAREST)

# Display the image in a CV window
cv2.imshow('Binary Image', resized_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()