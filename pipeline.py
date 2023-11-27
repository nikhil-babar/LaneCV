import cv2
from Callibration import CameraCalibration
from PerspectiveTransform import PerspectiveTransformation
from GradientThresholding import GradientThresholding
from LaneDetection import LaneDetection
from ImageDenoising import ImageDenoising
from ColorThresholding import ColorThresholding

# Set up the video capture
video_path = 'test_images\project_video.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))  # 3 corresponds to CV_CAP_PROP_FRAME_WIDTH
frame_height = int(cap.get(4))  # 4 corresponds to CV_CAP_PROP_FRAME_HEIGHT

threshold_range = (0.09, 0.6)
cal_pattern = (9, 6)
com_wt = (1, 0.6)

# Create instances of the required classes
cal = CameraCalibration(image_dir='camera_cal', pat_x=cal_pattern[0], pat_y=cal_pattern[1])
prst = PerspectiveTransformation(src_size=(frame_height, frame_width), dst_size=(frame_height, frame_width))
th = GradientThresholding(threshold_range=threshold_range)
detector = LaneDetection()
denoise = ImageDenoising()
color_filter = ColorThresholding()

# Set up video writer with MP4 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))
colour_thresholded = cv2.VideoWriter('colour_thresholded.mp4', fourcc, 20.0, (frame_width, frame_height))
birds_eye_view_video = cv2.VideoWriter('bird_eye_view.mp4', fourcc, 20.0, (frame_width, frame_height))
gradient_2d = cv2.VideoWriter('gradient_2d.mp4', fourcc, 20.0, (frame_width, frame_height))
detected_2d_lane = cv2.VideoWriter('detected_2d_lane.mp4', fourcc, 20.0, (frame_width, frame_height))
detected_3d_lane = cv2.VideoWriter('detected_3d_lane.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    # Bilateral filtering
    frame = denoise.apply_bilateral_filter(frame, 9, 75, 75)

    # Color Thresholding the image
    frame = color_filter.apply_color_threshold(frame)
    colour_thresholded.write(frame)

    undistorted_frame = cal.undistort(frame)

    # Apply perspective transformation
    birds_eye_view = prst.forward(undistorted_frame)
    birds_eye_view_video.write(birds_eye_view)

    # Apply gradient thresholding
    birds_eye_view_th = th.forward(birds_eye_view)
    gradient_2d.write(birds_eye_view_th)

    # Detect lanes
    birds_eye_view_detected_lane = detector.fit_polynomial(birds_eye_view_th)
    detected_2d_lane.write(birds_eye_view_detected_lane)

    # Reverse perspective transformation
    detected_lane = prst.backward(birds_eye_view_detected_lane)
    detected_3d_lane.write(detected_lane)

    # Display the results
    result = cv2.addWeighted(original_frame, com_wt[0], detected_lane, com_wt[1], 0)
    output_video.write(result)

    # Show progress in the console
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    progress = f"Progress: {current_frame}/{frame_count} frames processed"
    print(progress, end='\r')

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer, video capture object, and close windows
output_video.release()
colour_thresholded.release()
birds_eye_view_video.release()
gradient_2d.release()
detected_2d_lane.release()
detected_3d_lane.release
cap.release()
cv2.destroyAllWindows()

print("\nVideo processing complete. Output video saved as 'output_video.mp4'")
