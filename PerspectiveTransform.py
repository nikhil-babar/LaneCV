import numpy as np
import cv2

class PerspectiveTransformation:
    def __init__(self, src_size=(720,1280), dst_size=(720, 1280),
                 src=np.float32([(0.38, 0.62), (0.62, 0.62), (0.0, 0.9), (1, 0.9)]),
                 dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        self.src_size = np.float32([(src_size[1], src_size[0])])
        self.src = src * self.src_size
        self.dst_size = np.float32([(dst_size[1], dst_size[0])])
        self.dst = dst * self.dst_size
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=flags)
        sharpened = cv2.Laplacian(warped, cv2.CV_64F)
        sharpened = np.uint8(np.clip(warped - 0.5 * sharpened, 0, 255))

        return sharpened
    
    def backward(self, img, flags=cv2.INTER_LINEAR):
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        unwarped = cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]), flags=flags)

        return unwarped
