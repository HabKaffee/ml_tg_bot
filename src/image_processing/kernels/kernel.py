from typing import Protocol, List
import numpy as np


class Kernel(Protocol):
    """
    Kernel interface for image processing
    """

    def process(image: np.ndarray, params: List[str]) -> np.ndarray:
        """
        Process the given image using the specified parameters.

        Args:
            image (np.ndarray): The image to be processed, represented as a NumPy array.
            params (List[str]): A list of parameters required for processing the image.

        Returns:
            np.ndarray: The processed image, represented as a NumPy array.
        """
        ...
