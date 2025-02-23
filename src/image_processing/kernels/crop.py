import cv2
import numpy as np

from src.image_processing.command import CommandParameters
from src.image_processing.kernels.kernel import Kernel


class CropImage(Kernel):
    """
    Crop image
    """

    @staticmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Crop image
        Args:
            image: np.ndarray image
            params: CommandParameters
        Returns:
            image: np.ndarray image
        """
        width, height = int(params.width), int(params.height)
        x, y = image.shape[1] // 2, image.shape[0] // 2
        image = cv2.getRectSubPix(image, (width, height), (x, y))
        return image
