from typing import List

import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class SharpenImage(Kernel):
    """
    Sharpen image
    """

    @staticmethod
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Sharpen image
        Args:
            image: np.ndarray image
            params: list of parameters
        Returns:
            image: np.ndarray image
        """
        kernel = -1 * np.ones((3, 3))
        kernel[1, 1] = 9 + int(params[0])
        image = cv2.filter2D(image, -1, kernel)
        return image
