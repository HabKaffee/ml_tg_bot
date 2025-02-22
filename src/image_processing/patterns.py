from enum import Enum


class CommandType(Enum):
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


class CommandTypePatterns:
    def get_patterns(self):
        return {
            CommandType.RESIZE: r"изменить размер до (\d+)x(\d+)",
            CommandType.ROTATE: r"повернуть на (-?\d+) градусов",
            CommandType.FLIP: r"отразить (горизонтально|вертикально)",
            CommandType.BRIGHTNESS: r"изменить яркость на (-?\d+)",
            CommandType.CONTRAST: r"изменить контрастность на (-?\d+)",
            CommandType.CROP: r"обрезать до (\d+)x(\d+)",
            CommandType.GRAYSCALE: r"преобразовать в черно-белый",
            CommandType.INVERT: r"инвертировать цвета|инвертировать",
            CommandType.BLUR: r"размытие с радиусом (\d+)",
            CommandType.SHARPEN: r"повысить резкость на (-?\d+)",
            CommandType.GAMMA: r"изменить гамму на (-?\d+)",
        }
