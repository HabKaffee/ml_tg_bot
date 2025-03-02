import json
import re
from typing import Dict, List

from src.image_processing.command import Command
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def _get_llm_output(self, str: str) -> str:
        return """
        [
            {
                "action": "rotate",
                "parameters": {
                    "angle": 90
                }
            },
            {
                "action": "resize",
                "parameters": {
                    "width": 100,
                    "height": 100
                }
            }
        ]
        """

    def parse_text(self, text: str) -> List[Command]:

        processed_commands = self._get_llm_output(text)
        json_parameters = json.loads(processed_commands)

        commands = []
        for command in json_parameters:
            if command["action"] == "rotate":
                angle = command["parameters"]["angle"]
                commands.append(Command(KernelTypes.ROTATE, angle))
            elif command["action"] == "resize":
                width = command["parameters"]["width"]
                height = command["parameters"]["height"]
                commands.append(Command(KernelTypes.RESIZE, width, height))

        return commands
