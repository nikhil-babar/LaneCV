import cv2
import numpy as np

class GradientThresholding:
    def __init__(self, threshold_range=(0.2,0.5)):
        self.threshold_range = threshold_range

    def threshold_rel(self, img, lo, hi):
        vmin = np.min(img)
        vmax = np.max(img)

        vlo = vmin + (vmax - vmin) * lo
        vhi = vmin + (vmax - vmin) * hi
        return np.uint8((img >= vlo) & (img <= vhi)) * 255

    def forward(self, img):
        # Convert the image to HLS colorspace
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # Extract saturation and lightness channels
        saturation = hls[:, :, 2]
        lightness = hls[:, :, 1]

        # Apply Sobel in the x direction to saturation and lightness channels
        sobel_x_saturation = cv2.Sobel(saturation, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_lightness = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=3)

        # Combine Sobel operators
        sobel_x_combined = np.sqrt(sobel_x_saturation**2 + sobel_x_lightness**2)

        # Normalize to 8-bit for thresholding
        gradient_magnitude = np.uint8(255 * sobel_x_combined / np.max(sobel_x_combined))

        # Apply thresholding
        thresholded_gradient = self.threshold_rel(gradient_magnitude, self.threshold_range[0], self.threshold_range[1])

        return thresholded_gradient
