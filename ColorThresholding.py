import cv2
import os
import numpy as np

class ColorThresholding:
    def __init__(self):
        pass

    def apply_color_threshold(self, image):
        # Convert the image to the HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        # Define white color range in HLS
        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        # Define yellow color range in HLS
        lower_yellow = np.array([20, 120, 80], dtype=np.uint8)
        upper_yellow = np.array([45, 200, 255], dtype=np.uint8)

        # Threshold the image to get white and yellow lane pixels
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        # Combine the white and yellow masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=combined_mask)

        return result