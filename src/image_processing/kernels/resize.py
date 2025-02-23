import cv2
import numpy as np

from src.image_processing.command import CommandParameters
from src.image_processing.kernels.kernel import Kernel


class ResizeImage(Kernel):
    """
    Resize image
    """

    @staticmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Resize image
        Args:
            image: np.ndarray image
            params: CommandParameters
        Returns:
            image: np.ndarray image
        """
        width, height = int(params.width), int(params.height)
        return cv2.resize(image, (width, height))
