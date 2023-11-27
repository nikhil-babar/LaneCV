import numpy as np
import cv2
import os

class CameraCalibration():
    def __init__(self, image_dir, pat_x, pat_y):
        fnames = os.listdir(image_dir)
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((pat_x*pat_y, 3), np.float32)
        objp[:,:2] = np.mgrid[0:pat_x, 0:pat_y].T.reshape(-1, 2)
        
        for f in fnames:
            img = cv2.imread(os.path.join(image_dir, f))
            ret, corners = cv2.findChessboardCorners(img, (pat_x, pat_y))
            if ret:
                print(f'{f} processed')
                imgpoints.append(corners)
                objpoints.append(objp)
            else :
                print(f'{f} processing error')
 
        y, x, _ = cv2.imread(os.path.join(image_dir, fnames[0])).shape
        self.ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (y,x), None, None)

        if not self.ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)



