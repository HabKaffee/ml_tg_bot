import re
from typing import List, Tuple
from .patterns import CommandsPatterns, ImageProcessCommands

class ImageCommandsParser:
    def __init__(self, text: str):
        self.text = text.lower()
        self.commands = self._parse_commands()

    def _parse_commands(self) -> List[Tuple[str, List[str]]]:
        patterns = CommandsPatterns().get_commands()
        commands = []
        for command, data in patterns.items():
            pattern = data[ImageProcessCommands.PATTERN]
            function = data[ImageProcessCommands.COMMAND]
            matches = re.findall(pattern, self.text)
            for match in matches:
                if isinstance(match, tuple):
                    commands.append((command, list(match), function))
                else:
                    commands.append((command, [match], function))
        return commands
    
    def get_commands(self) -> List[Tuple[str, List[str]]]:
        return self.commands
