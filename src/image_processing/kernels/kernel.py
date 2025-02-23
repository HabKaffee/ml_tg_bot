from abc import ABC, abstractmethod

import numpy as np

from src.image_processing.command import CommandParameters


class Kernel(ABC):
    """
    Kernel interface for image processing
    """

    @staticmethod
    @abstractmethod
    def process(image: np.ndarray, params: CommandParameters) -> np.ndarray:
        """
        Process the given image using the specified parameters.

        Args:
            image (np.ndarray): The image to be processed, represented as a NumPy array.
            params: CommandParameters

        Returns:
            np.ndarray: The processed image, represented as a NumPy array.
        """
