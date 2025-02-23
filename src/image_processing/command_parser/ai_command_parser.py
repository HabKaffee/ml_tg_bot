import re
from typing import Dict, List

from src.image_processing.command import Command
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def _get_prompt(self) -> str:
        supported_kernels = ", ".join([f"{kernel.value}" for kernel in KernelTypes])

        prompt = f"""
        You are an advanced image editing assistant.
        Your task is to analyze the user's input and generate a clear, concise list of specific commands for editing an image.
        The commands should include {supported_kernels}. Ensure the commands are precise and actionable.

        For example:
        User Input: 'Make the image smaller and rotate it 90 degrees to the right.'
        Your Output:
            1	Adjust brightness: Increase by 20%.
            2	Rotate: 90 degrees clockwise.
        User Input: 'Flip the image horizontally and make the colors more vibrant.'
        Your Output:
            1	Flip: Horizontally.
            2	Adjust saturation: Increase by 30%.
        Now, analyze the following user input and generate the corresponding list of commands:
        """
        return prompt

    def parse_text(self, text: str) -> List[Command]:
        prompt = self._get_prompt()

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
