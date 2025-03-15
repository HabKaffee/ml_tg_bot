from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from src.image_processing.command import Command


class CommandParser(ABC):
    """
    Interface for command parser
    """

    @abstractmethod
    def parse_text(self, text: str) -> List[Command]:
        """
        Parse text
        Args:
            text (str): text to parse
        """

    @abstractmethod
    def analyze_image(self, image: Image.Image) -> None:
        """
        Parse image
        Args:
            image (Image.Image): image to parse
        """
