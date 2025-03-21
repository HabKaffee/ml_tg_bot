import re
from typing import Dict, List

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser, ParserParameters
from src.image_processing.kernels.kernel_types import KernelTypes


class PatternCommandParser(CommandParser):
    """
    Command parser based on patterns
    """

    def _get_patterns(self) -> Dict[KernelTypes, str]:
        return {
            KernelTypes.BLUR: r"размытие с радиусом (?P<step>-?\d+)",
            KernelTypes.CONTRAST: r"изменить контрастность на (?P<step>-?\d+)",
            KernelTypes.CROP: r"обрезать до (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.GRAYSCALE: r"преобразовать в черно-белый",
            KernelTypes.INVERT: r"инвертировать цвета|инвертировать",
            KernelTypes.RESIZE: r"изменить размер до (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.ROTATE: r"повернуть на (?P<angle>-?\d+) градусов",
            KernelTypes.SHARPEN: r"повысить резкость на (?P<step>-?\d+)",
        }

    def parse_text(self, text: str, _: ParserParameters) -> List[Command]:
        text = text.lower()

        patterns = self._get_patterns()
        commands_list = []
        for command_type, pattern in patterns.items():
            match = re.match(pattern, text)
            if match is None:
                continue

            parameters = CommandParameters(**match.groupdict())
            command = Command(command_type, parameters)

            commands_list.append(command)
        return commands_list
