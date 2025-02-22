from abc import ABC, abstractmethod
from typing import List

from src.image_processing.command import Command


class CommandParser(ABC):
    """
    Interface for command parser
    """

    @abstractmethod
    def parse_text(self, text: str) -> List[Command]:
        """
        Parse text to commands
        Args:
            text: input text
        Returns:
            commands: list of commands
        """
