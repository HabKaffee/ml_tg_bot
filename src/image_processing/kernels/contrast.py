from typing import List

import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class ContrastImage(Kernel):
    """
    Change image contrast
    """

    @staticmethod
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Change image contrast
        Args:
            image: np.ndarray image
            params: list of parameters
        Returns:
            image: np.ndarray image
        """
        alpha = 1.0 + int(params[0]) / 100.0
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return image
