from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from src.image_processing.command import Command
from src.image_processing.command_parser.language_package import LanguageType, get_language_package


@dataclass
class ParserParameters:
    """
    Parameters for command parser
    """

    num_few_shot_samples: int = -1
    analyze_image: bool = False
    image_to_analyze: Optional[Image.Image] = None


class CommandParser(ABC):
    """
    Interface for command parser
    """

    def __init__(self, language: LanguageType = LanguageType.EN):
        super().__init__()
        self._language = language
        self._language_package = get_language_package(self._language)

    @abstractmethod
    def parse_text(self, text: str, parser_parameters: ParserParameters) -> List[Command]:
        """
        Parse text
        Args:
            text (str): text to parse
            parser_parameters (ParserParameters): parameters for parser
        Returns:
            List[Command]: list of commands
        """
