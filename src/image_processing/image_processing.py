from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .mapping import ImageCommandsParser


class ImageProcessor(ABC):
    @abstractmethod
    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image: ...

    @abstractmethod
    def _apply_commands(self, image, commands: List[Tuple[str, List[str], Callable]]) -> Image.Image: ...


class ImageProcessorByText(ImageProcessor):
    def _apply_commands(self, image, commands: List[Tuple[str, List[str], Callable]]) -> Image.Image:
        for command, params, func in commands:
            match command:
                case "resize":
                    width, height = map(int, params)
                    image = func(image, (width, height))
                case "rotate":
                    rotations = {90: cv2.ROTATE_90_CLOCKWISE, -90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180}
                    if (angle := int(params[0])) in rotations:
                        image = func(image, rotations[angle])
                case "flip":
                    mode = 1 if params[0] == "горизонтально" else 0
                    image = func(image, mode)
                case _ if command in ("brightness", "contrast"):
                    alpha = 1.0 + int(params[0]) / 100.0
                    image = func(image, alpha=alpha, beta=0)
                case "crop":
                    width, height = map(int, params)
                    x, y = image.shape[1] // 2, image.shape[0] // 2
                    image = func(image, (width, height), (x, y))
                case "grayscale":
                    image = func(image, cv2.COLOR_BGR2GRAY)
                case "invert":
                    image = func(image)
                case "blur":
                    ksize = int(params[0])
                    image = func(image, (ksize, ksize), 0)
                case "sharpen":
                    kernel = -1 * np.ones((3, 3))
                    kernel[1, 1] = 9 + int(params[0])
                    image = func(image, -1, kernel)
                case "saturation" | "hue":
                    image = func(image, cv2.COLOR_BGR2HSV)
                case "gamma":
                    gamma = float(params[0])
                    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
                    image = func(image, table)
        return image

    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image:
        commands = ImageCommandsParser(text).get_commands()
        return self._apply_commands(image, commands)
