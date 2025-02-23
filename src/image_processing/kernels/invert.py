import cv2
import numpy as np

from src.image_processing.command import CommandParameters
from src.image_processing.kernels.kernel import Kernel


class InvertImage(Kernel):
    """
    Invert image colors
    """

    @staticmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Invert image colors
        Args:
            image: np.ndarray image
            params: CommandParameters
        Returns:
            image: np.ndarray image
        """
        image = cv2.bitwise_not(image)
        return image
