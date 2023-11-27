import cv2
import os
import numpy as np

# img = cv2.imread('test_images\challenge_video_frame_1.jpg')

# scale = np.float32([(0.38, 0.62), (0.62, 0.62), (1, 0.9), (0.0, 0.9)])
# img_size = np.float32([(img.shape[1], img.shape[0])])

# # Calculate coordinates for the parallelogram
# co = scale * img_size
# co = co.astype(int)  # Convert to integers for cv2.polylines

# # Reshape the coordinates for cv2.polylines
# points = co.reshape((-1, 1, 2))

# # Draw the parallelogram on the image
# cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

# # Display the image with the drawn parallelogram
# cv2.imshow('Image with Parallelogram', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread('test_images/test4.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with the actual image path

# # Check if the image is successfully loaded
# if img is None:
#     print("Error: Could not read the image.")
#     exit()

# # Calculate the gradient in the x-direction
# gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# # Calculate the gradient in the y-direction
# gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# # Calculate the magnitude of the gradient
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# # Display the original image and the gradient images
# cv2.imshow('Gradient Magnitude', cv2.convertScaleAbs(gradient_magnitude))

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# class GradientThresholding:
#     def __init__(self):
#         pass

#     def threshold_rel(self, img, lo, hi):
#         vmin = np.min(img)
#         vmax = np.max(img)
        
#         vlo = vmin + (vmax - vmin) * lo
#         vhi = vmin + (vmax - vmin) * hi
#         return np.uint8((img >= vlo) & (img <= vhi)) * 255

#     def forward(self, img):
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         # Apply Sobel in the x and y directions
#         sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

#         # Calculate the magnitude of the gradient
#         gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

#         # Normalize the magnitude to the range [0, 255]
#         gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

#         # Apply thresholding to the magnitude
#         thresholded_gradient = self.threshold_rel(gradient_magnitude, 0.2, 1.0)

#         return thresholded_gradient
    
# class Thresholding:
#     def __init__(self):
#         pass
    
#     def threshold_rel(self, img, lo, hi):
#         vmin = np.min(img)
#         vmax = np.max(img)
        
#         vlo = vmin + (vmax - vmin) * lo
#         vhi = vmin + (vmax - vmin) * hi
#         return np.uint8((img >= vlo) & (img <= vhi)) * 255

#     def threshold_abs(self, img, lo, hi):
#         return np.uint8((img >= lo) & (img <= hi)) * 255

#     def forward(self, img):
#         hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         h_channel = hls[:,:,0]
#         l_channel = hls[:,:,1]
#         v_channel = hsv[:,:,2]

#         right_lane = self.threshold_rel(l_channel, 0.8, 1.0)
#         right_lane[:,:750] = 0

#         left_lane = self.threshold_abs(h_channel, 20, 30)
#         left_lane &= self.threshold_rel(v_channel, 0.8, 1.0)
#         left_lane[:,550:] = 0

#         img2 = left_lane | right_lane

#         return img2

# # Example usage:
# img = cv2.imread('test_images\project_video_frame_1032.jpg')  # Replace with the actual image path

# img_copy = img.copy()

# # Using the modified GradientThresholding class
# gradient_thresholding = GradientThresholding()
# gradient_thresholded_img = gradient_thresholding.forward(img)

# normal_thresholding = Thresholding()
# normal_thresholded_img = normal_thresholding.forward(img_copy)

# # Display the original and gradient-thresholded images
# cv2.imshow('Original Image', img)
# cv2.imshow('Gradient Thresholded Image', gradient_thresholded_img)
# cv2.imshow('Normal Thresholded Image', normal_thresholded_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# class ColorThresholding:
#     def __init__(self):
#         pass

#     def apply_color_threshold(self, image):
#         # Convert the image to the HLS color space
#         hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

#         # Define white color range in HLS
#         lower_white = np.array([0, 200, 0], dtype=np.uint8)
#         upper_white = np.array([255, 255, 255], dtype=np.uint8)

#         # Define yellow color range in HLS
#         lower_yellow = np.array([20, 120, 80], dtype=np.uint8)
#         upper_yellow = np.array([45, 200, 255], dtype=np.uint8)

#         # Threshold the image to get white and yellow lane pixels
#         white_mask = cv2.inRange(hls, lower_white, upper_white)
#         yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

#         # Combine the white and yellow masks
#         combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

#         # Apply the mask to the original image
#         result = cv2.bitwise_and(image, image, mask=combined_mask)

#         return result
    
# input_image = cv2.imread("test_images\challenge_video_frame_110.jpg")

# # Create an instance of LaneDetector
# lane_detector = ColorThresholding()

# # Apply color thresholding
# result_image = lane_detector.apply_color_threshold(input_image)

# # Display the original and result images
# cv2.imshow("Original Image", input_image)
# cv2.imshow("Result Image", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()