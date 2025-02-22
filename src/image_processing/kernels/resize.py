from typing import List
import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class ResizeImage(Kernel):
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        width, height = map(int, params)
        return cv2.resize(image, (width, height))
