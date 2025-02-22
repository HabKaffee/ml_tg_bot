import re
from typing import Dict, List

from src.image_processing.command import Command
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


class PatternCommandParser(CommandParser):
    """
    Command parser based on patterns
    """

    def _get_patterns(self) -> Dict[KernelTypes, str]:
        return {
            KernelTypes.RESIZE: r"изменить размер до (\d+)x(\d+)",
            KernelTypes.ROTATE: r"повернуть на (-?\d+) градусов",
            KernelTypes.CONTRAST: r"изменить контрастность на (-?\d+)",
            KernelTypes.CROP: r"обрезать до (\d+)x(\d+)",
            KernelTypes.GRAYSCALE: r"преобразовать в черно-белый",
            KernelTypes.INVERT: r"инвертировать цвета|инвертировать",
            KernelTypes.BLUR: r"размытие с радиусом (\d+)",
            KernelTypes.SHARPEN: r"повысить резкость на (-?\d+)",
        }

    def parse_text(self, text: str) -> List[Command]:
        text = text.lower()

        patterns = self._get_patterns()
        commands_list = []
        for command_type, pattern in patterns.items():
            matches = re.findall(pattern, text)

            for match in matches:
                if isinstance(match, tuple):
                    command = Command(command_type, list(match))
                else:
                    command = Command(command_type, [match])

                commands_list.append(command)
        return commands_list
