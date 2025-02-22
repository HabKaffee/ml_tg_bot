from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image
from src.image_processing.kernels.resize import ResizeImage
from src.image_processing.kernels.rotate import RotateImage
from utils import cv2_to_pil, pil_to_cv2

from src.image_processing.mapping import Command, ImageCommandsParser
from src.image_processing.patterns import Commands


class ImageProcessor(ABC):
    @abstractmethod
    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image: ...

    @abstractmethod
    def _apply_commands(self, image: np.ndarray, commands: List[Tuple[str, List[str], Callable]]) -> np.ndarray: ...


command_matching = {
    Commands.RESIZE: ResizeImage,
    Commands.ROTATE: RotateImage,
}


class ImageProcessorByText(ImageProcessor):
    def _apply_commands(self, image: np.ndarray, commands: List[Command]) -> np.ndarray:
        for command in commands:
            if command.command_type in command_matching:
                image = command_matching[command.command_type].process(
                    image=image,
                    params=command.parameters
                )
        return image

                # case Commands.FLIP:
                # Should be implemented without text
                #     mode = 1 if params[0] == "горизонтально" else 0
                #     image = cv2.flip(image, mode)

                # case Commands.BRIGHTNESS | Commands.CONTRAST:
                #     alpha = 1.0 + int(params[0]) / 100.0
                #     image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

                # case Commands.CROP:
                #     width, height = map(int, params)
                #     x, y = image.shape[1] // 2, image.shape[0] // 2
                #     image = cv2.getRectSubPix(image, (width, height), (x, y))

                # case Commands.GRAYSCALE:
                #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # case Commands.INVERT:
                #     image = cv2.bitwise_not(image)

                # case Commands.BLUR:
                #     ksize = int(params[0])
                #     image = cv2.GaussianBlur(image, (ksize, ksize), 0)

                # case Commands.SHARPEN:
                #     kernel = -1 * np.ones((3, 3))
                #     kernel[1, 1] = 9 + int(params[0])
                #     image = cv2.filter2D(image, -1, kernel)

                # case Commands.GAMMA:
                #     gamma = float(params[0])
                #     table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
                #     image = cv2.LUT(image, table)

    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image:
        commands = ImageCommandsParser(text).get_commands()

        inner_image_representation = pil_to_cv2(image)
        inner_image_representation = self._apply_commands(inner_image_representation, commands)

        return cv2_to_pil(inner_image_representation)
