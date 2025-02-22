from typing import List
import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class RotateImage(Kernel):
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        rotations = {
            90: cv2.ROTATE_90_CLOCKWISE,
            -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180
        }

        angle = int(params[0])

        if angle not in rotations:
            print("Invalid angle")
            return image

        return cv2.rotate(image, rotations[angle])
