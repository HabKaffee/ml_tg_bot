from dataclasses import dataclass
import re
from typing import List

from src.image_processing.patterns import Commands, CommandsPatterns, ImageProcessCommands


@dataclass
class Command:
    def __init__(self, command_type: Commands, parameters: List[str]):
        self.command_type = command_type
        self.parameters = parameters


class ImageCommandsParser:
    def __init__(self, text: str):
        self.text = text.lower()
        self.commands = self._parse_commands()

    def _parse_commands(self) -> List[Command]:
        patterns = CommandsPatterns().get_commands()
        commands = []
        for command, data in patterns.items():
            pattern = data[ImageProcessCommands.PATTERN]
            matches = re.findall(pattern, self.text)

            for match in matches:
                if isinstance(match, tuple):
                    command = Command(command, list(match))
                else:
                    command = Command(command, [match])

                commands.append(command)
        return commands

    def get_commands(self) -> List[Command]:
        return self.commands
