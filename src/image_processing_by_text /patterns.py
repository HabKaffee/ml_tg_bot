from typing import Dict


class CommandsPatterns:
    PATTERNS = {
        "resize": r"изменить размер до (\d+)x(\d+)",
        "rotate": r"повернуть на (-?\d+) градусов",
        "flip": r"отразить (горизонтально|вертикально)",
        "brightness": r"изменить яркость на (-?\d+)",
        "contrast": r"изменить контрастность на (-?\d+)",
        "crop": r"обрезать до (\d+)x(\d+)",
        "grayscale": r"преобразовать в черно-белый",
        "invert": r"инвертировать цвета",
        "blur": r"размытие с радиусом (\d+)",
        "sharpen": r"повысить резкость на (-?\d+)",
        "saturation": r"изменить насыщенность на (-?\d+)",
        "hue": r"изменить оттенок на (-?\d+) градусов",
        "gamma": r"изменить гамму на (-?\d+)",
        "add_text": r"добавить текст \"(.+)\" в позицию (\d+),(\d+)"
    }

    @classmethod
    def get_patterns(cls) -> Dict[str, str]:
        return cls.PATTERNS
