import logging
from typing import List

import numpy as np
from PIL import Image

from src.image_processing.command_parser.command_parser import Command
from src.image_processing.kernels.kernel_map import get_kernel_map
from src.image_processing.utils import cv2_to_pil, pil_to_cv2


class ImageProcessor:
    """
    ImageProcessor class.
    """

    def __init__(self) -> None:
        self._kernel_map = get_kernel_map()

    def get_processed_image(self, image: Image.Image, command_queue: List[Command]) -> Image.Image:
        """
        Get processed image by text
        Args:
            image: PIL opened image
            text: text command
        Returns:
            image: PIL opened image
        """
        inner_image_representation = pil_to_cv2(image)
        inner_image_representation = self._apply_commands(inner_image_representation, command_queue)

        output_image = cv2_to_pil(inner_image_representation)
        return output_image

    def _apply_commands(self, image: np.ndarray, command_queue: List[Command]) -> np.ndarray:
        """
        Apply commands to the image
        Args:
            image: np.ndarray image
            command_queue: list of commands
        Returns:
            image: np.ndarray image
        """
        for command in command_queue:
            if command.kernel_type in self._kernel_map:
                logging.info("Applying command: %s with parameters: %s", command.kernel_type, command.parameters)
                image = self._kernel_map[command.kernel_type].process(image=image, params=command.parameters)
        return image
