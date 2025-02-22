from enum import Enum

import cv2


class ImageProcessCommands(Enum):
    PATTERN = "pattern"
    COMMAND = "command"


class Commands(Enum):
    RESIZE = "resize"
    ROTATE = "rotate"
    FLIP = "flip"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    CROP = "crop"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    BLUR = "blur"
    SHARPEN = "sharpen"
    SATURATION = "saturation"
    HUE = "hue"
    GAMMA = "gamma"


class CommandsPatterns:
    def get_commands(self):
        return {
            Commands.RESIZE: {
                ImageProcessCommands.PATTERN: r"изменить размер до (\d+)x(\d+)",
            },
            Commands.ROTATE: {
                ImageProcessCommands.PATTERN: r"повернуть на (-?\d+) градусов",
            },
            Commands.FLIP: {
                ImageProcessCommands.PATTERN: r"отразить (горизонтально|вертикально)",
            },
            Commands.BRIGHTNESS: {
                ImageProcessCommands.PATTERN: r"изменить яркость на (-?\d+)",
            },
            Commands.CONTRAST: {
                ImageProcessCommands.PATTERN: r"изменить контрастность на (-?\d+)",
            },
            Commands.CROP: {
                ImageProcessCommands.PATTERN: r"обрезать до (\d+)x(\d+)",
            },
            Commands.GRAYSCALE: {
                ImageProcessCommands.PATTERN: r"преобразовать в черно-белый",
            },
            Commands.INVERT: {
                ImageProcessCommands.PATTERN: r"инвертировать цвета|инвертировать",
            },
            Commands.BLUR: {
                ImageProcessCommands.PATTERN: r"размытие с радиусом (\d+)",
            },
            Commands.SHARPEN: {
                ImageProcessCommands.PATTERN: r"повысить резкость на (-?\d+)",
            },
            Commands.GAMMA: {
                ImageProcessCommands.PATTERN: r"изменить гамму на (-?\d+)",
            },
        }
