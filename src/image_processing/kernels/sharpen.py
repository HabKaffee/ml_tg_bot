import cv2
import numpy as np

from src.image_processing.command import CommandParameters
from src.image_processing.kernels.kernel import Kernel


class SharpenImage(Kernel):
    """
    Sharpen image
    """

    @staticmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Sharpen image
        Args:
            image: np.ndarray image
            params: CommandParameters
        Returns:
            image: np.ndarray image
        """
        kernel = -1 * np.ones((3, 3))
        kernel[1, 1] = 9 + int(params.step)
        image = cv2.filter2D(image, -1, kernel)
        return image
