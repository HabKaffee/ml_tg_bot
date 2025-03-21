import re
from typing import List

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser, ParserParameters


class PatternCommandParser(CommandParser):
    """
    Command parser based on patterns
    """

    def parse_text(self, text: str, _: ParserParameters) -> List[Command]:
        text = text.lower()

        patterns = self._language_package.get_patters()
        commands_list = []
        for command_type, pattern in patterns.items():
            match = re.match(pattern, text)
            if match is None:
                continue

            parameters = CommandParameters(**match.groupdict())
            command = Command(command_type, parameters)

            commands_list.append(command)
        return commands_list
