import cv2

class ImageDenoising:
    def __init__(self):
        pass

    def apply_bilateral_filter(self, input_image, d, sigma_color, sigma_space):
        # Apply bilateral filter to the input image
        filtered_image = cv2.bilateralFilter(input_image, d, sigma_color, sigma_space)

        return filtered_image