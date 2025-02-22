from enum import Enum

import cv2


class ImageProcessCommands(Enum):
    PATTERN = "pattern"
    COMMAND = "command"


class CommandsPatterns:
    def get_commands(self):
        return {
            "resize": {
                ImageProcessCommands.PATTERN: r"изменить размер до (\d+)x(\d+)",
                ImageProcessCommands.COMMAND: cv2.resize,
            },
            "rotate": {
                ImageProcessCommands.PATTERN: r"повернуть на (-?\d+) градусов",
                ImageProcessCommands.COMMAND: cv2.rotate,
            },
            "flip": {
                ImageProcessCommands.PATTERN: r"отразить (горизонтально|вертикально)",
                ImageProcessCommands.COMMAND: cv2.flip,
            },
            "brightness": {
                ImageProcessCommands.PATTERN: r"изменить яркость на (-?\d+)",
                ImageProcessCommands.COMMAND: cv2.convertScaleAbs,
            },
            "contrast": {
                ImageProcessCommands.PATTERN: r"изменить контрастность на (-?\d+)",
                ImageProcessCommands.COMMAND: cv2.convertScaleAbs,
            },
            "crop": {
                ImageProcessCommands.PATTERN: r"обрезать до (\d+)x(\d+)",
                ImageProcessCommands.COMMAND: cv2.getRectSubPix,
            },
            "grayscale": {
                ImageProcessCommands.PATTERN: r"преобразовать в черно-белый",
                ImageProcessCommands.COMMAND: cv2.cvtColor,
            },
            "invert": {
                ImageProcessCommands.PATTERN: r"инвертировать цвета|инвертировать",
                ImageProcessCommands.COMMAND: cv2.bitwise_not,
            },
            "blur": {
                ImageProcessCommands.PATTERN: r"размытие с радиусом (\d+)",
                ImageProcessCommands.COMMAND: cv2.GaussianBlur,
            },
            "sharpen": {
                ImageProcessCommands.PATTERN: r"повысить резкость на (-?\d+)",
                ImageProcessCommands.COMMAND: cv2.filter2D,
            },
            "saturation": {
                ImageProcessCommands.PATTERN: r"изменить насыщенность на (-?\d+)",
                ImageProcessCommands.COMMAND: cv2.cvtColor,
            },
            "hue": {
                ImageProcessCommands.PATTERN: r"изменить оттенок на (-?\d+) градусов",
                ImageProcessCommands.COMMAND: cv2.cvtColor,
            },
            "gamma": {
                ImageProcessCommands.PATTERN: r"изменить гамму на (-?\d+)",
                ImageProcessCommands.COMMAND: cv2.LUT,
            },
        }
