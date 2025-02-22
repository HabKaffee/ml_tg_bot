from typing import List

import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class BlurImage(Kernel):
    """
    Blur image
    """

    @staticmethod
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Blur image
        Args:
            image: np.ndarray image
            params: list of parameters
        Returns:
            image: np.ndarray image
        """
        ksize = int(params[0])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image
