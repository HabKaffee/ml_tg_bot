import json
from typing import List

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def _get_llm_output(self, _: str) -> str:
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
            action = command["action"]
            parameters = command["parameters"]

            kernel_type = KernelTypes(action)
            parameters = CommandParameters(**parameters)

            commands.append(Command(kernel_type, parameters))

        return commands
