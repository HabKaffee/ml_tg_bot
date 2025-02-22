from typing import List

import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class GrayscaleImage(Kernel):
    """
    Convert image to grayscale
    """

    @staticmethod
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Convert image to grayscale
        Args:
            image: np.ndarray image
            params: list of parameters
        Returns:
            image: np.ndarray image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
