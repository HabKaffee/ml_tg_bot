import cv2
import numpy as np

from src.image_processing.command import CommandParameters
from src.image_processing.kernels.kernel import Kernel


class BlurImage(Kernel):
    """
    Blur image
    """

    @staticmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Blur image
        Args:
            image: np.ndarray image
            params: CommandParameters
        Returns:
            image: np.ndarray image
        """
        ksize = int(params.step)
        if ksize % 2 == 0:
            ksize += 1
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image
