import re
from dataclasses import dataclass
from typing import List

from src.image_processing.patterns import CommandType, CommandTypePatterns


@dataclass
class Command:
    def __init__(self, command_type: CommandType, parameters: List[str]):
        self.command_type = command_type
        self.parameters = parameters


class ImageCommandsParser:
    def __init__(self, text: str):
        self.text = text.lower()
        self._commands = self._parse_commands()

    def _parse_commands(self) -> List[Command]:
        patterns = CommandTypePatterns().get_patterns()
        commands = []
        for command_type, pattern in patterns.items():
            matches = re.findall(pattern, self.text)

            for match in matches:
                if isinstance(match, tuple):
                    command_type = Command(command_type, list(match))
                else:
                    command_type = Command(command_type, [match])

                commands.append(command_type)
        return commands

    def get_commands(self) -> List[Command]:
        return self._commands
