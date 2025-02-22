import re
from typing import List, Tuple
from patterns import CommandsPatterns

class ImageCommandParser:
    def __init__(self, text: str):
        self.text = text.lower()
        self.commands = self._parse_commands()
    
    def _parse_commands(self) -> List[Tuple[str, List[str]]]:
        commands_patterns = CommandsPatterns().get_patterns()
        commands = []
        for command, pattern in commands_patterns.items():
            matches = re.findall(pattern, self.text)
            for match in matches:
                if isinstance(match, tuple):
                    commands.append((command, list(match)))
                else:
                    commands.append((command, [match]))
        return commands
    
    def get_commands(self) -> List[Tuple[str, List[str]]]:
        return self.commands
