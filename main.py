import os
import cv2
from Callibration import CameraCalibration
from PerspectiveTransform import PerspectiveTransformation
from GradientThresholding import GradientThresholding
from LaneDetection import LaneDetection
from ImageDenoising import ImageDenoising
from ColorThresholding import ColorThresholding

cal_image_path = 'test_images'
filename = 'test_19.jpg'

cal = CameraCalibration(image_dir='camera_cal', pat_x=9, pat_y=6)

img = cv2.imread(os.path.join(cal_image_path, filename))
img = cv2.resize(img, (1280, 720))
original_img = img.copy()

img = cal.undistort(img=img)
if img is not None:
    cv2.imshow('image', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else:
    raise Exception("Unable to get callibrated image")

denoise = ImageDenoising()
d = 9  # Diameter of each pixel neighborhood
sigma_color = 75  # Filter sigma in the color space
sigma_space = 75  # Filter sigma in the coordinate space

# Apply bilateral filtering
img = denoise.apply_bilateral_filter(img, d, sigma_color, sigma_space)

# Display the original and result images
if img is not None:
    cv2.imshow('Denoised image', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to denoise image")

color_filter = ColorThresholding()
img = color_filter.apply_color_threshold(img)

if img is not None:
    cv2.imshow('Color thresholded image', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to apply color thresholding")

prst = PerspectiveTransformation(src_size=(img.shape[0], img.shape[1]), dst_size=(img.shape[0], img.shape[1]))
birds_eye_view = prst.forward(img)

if birds_eye_view is not None:
    cv2.imshow('Top view of image', birds_eye_view)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to obtain the birds eye view")

th = GradientThresholding(threshold_range=(0.09, 0.6))
birds_eye_view_th = th.forward(birds_eye_view)

if birds_eye_view_th is not None:
    cv2.imshow('Binary image', birds_eye_view_th)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to obtain the threshoulded binary image")

detector = LaneDetection()
birds_eye_view_detected_lane = detector.fit_polynomial(birds_eye_view_th)

if  birds_eye_view_detected_lane is not None:
    cv2.imshow('2d Lanes', birds_eye_view_detected_lane)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to detect bird's eye view lanes")

detected_lane = prst.backward(birds_eye_view_detected_lane)

if  detected_lane is not None:
    result = cv2.addWeighted(original_img, 1, detected_lane, 0.6, 0)
    cv2.imshow('Result', result)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
else: 
    raise Exception("Unable to detect lanes")

