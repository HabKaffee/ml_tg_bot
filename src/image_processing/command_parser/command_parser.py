from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, TypeAlias

from PIL import Image

from src.image_processing.command import Command
from src.image_processing.command_parser.language import LanguageType

LanguagePackage: TypeAlias = Dict[str, str]

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
