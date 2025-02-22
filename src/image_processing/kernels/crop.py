from typing import List

import cv2
import numpy as np

from src.image_processing.kernels.kernel import Kernel


class CropImage(Kernel):
    """
    Crop image
    """

    @staticmethod
    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Crop image
        Args:
            image: np.ndarray image
            params: list of parameters
        Returns:
            image: np.ndarray image
        """
        width, height = map(int, params)
        x, y = image.shape[1] // 2, image.shape[0] // 2
        image = cv2.getRectSubPix(image, (width, height), (x, y))
        return image
