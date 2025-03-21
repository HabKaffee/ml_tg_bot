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
        self.language = language
        self._language_package_path = Path(
            f"src/image_processing/command_parser/language_package_{self._language}.json"
        )
        self._language_package = self._load_language_package(self._language_package_path)

    @staticmethod
    def _load_language_package(language_package: Path) -> LanguagePackage:
        if not language_package.exists():
            raise FileNotFoundError(f"There is no file {language_package} with language package")

        try:
            with language_package.open(mode="r", encoding="utf-8") as file:
                return json.load(file)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f"Loading file with language package raised {exc} exception")

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
