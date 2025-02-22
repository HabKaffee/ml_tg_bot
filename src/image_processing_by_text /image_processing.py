from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np
from .mapping import ImageCommandsParser

class ImageProcessor(ABC):
    @abstractmethod
    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image: ...
    @abstractmethod
    def _apply_commands(self, image, commands: List[Tuple[str, List[str], callable]]): ...

class ImageProcessorByText(ImageProcessor):
    def _apply_commands(self, image, commands: List[Tuple[str, List[str], callable]]):
        for command, params, func in commands:
            if command == "resize":
                width, height = map(int, params)
                image = func(image, (width, height))
            elif command == "rotate":
                angle = int(params[0])
                if angle == 90:
                    image = func(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == -90:
                    image = func(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    image = func(image, cv2.ROTATE_180)
            elif command == "flip":
                mode = 1 if params[0] == "горизонтально" else 0
                image = func(image, mode)
            elif command in ("brightness", "contrast"):
                alpha = 1.0 + int(params[0]) / 100.0
                image = func(image, alpha=alpha, beta=0)
            elif command == "crop":
                width, height = map(int, params)
                x, y = image.shape[1] // 2, image.shape[0] // 2
                image = func(image, (width, height), (x, y))
            elif command == "grayscale":
                image = func(image, cv2.COLOR_BGR2GRAY)
            elif command == "invert":
                image = func(image)
            elif command == "blur":
                ksize = int(params[0])
                image = func(image, (ksize, ksize), 0)
            elif command == "sharpen":
                kernel = -1 * np.ones((3, 3))
                kernel[1, 1] = 9 + int(params[0])
                image = func(image, -1, kernel)
            elif command == "saturation":
                image = func(image, cv2.COLOR_BGR2HSV)
            elif command == "hue":
                image = func(image, cv2.COLOR_BGR2HSV)
                gamma = float(params[0])
                table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
                image = func(image, table)
        return image

    def get_processed_image(self, image: Image.Image, text: str) -> Image.Image:
        commands = ImageCommandsParser(text).get_commands()
        return  self._apply_commands(image, commands)
