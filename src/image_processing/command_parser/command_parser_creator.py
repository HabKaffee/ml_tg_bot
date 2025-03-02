from enum import Enum

from src.image_processing.command_parser.ai_command_parser import AICommandParser
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.command_parser.pattern_command_parser import PatternCommandParser


class CommandParserTypes(Enum):
    PATTERN = "pattern"
    AI = "ai"


def get_command_parser(command_parser: CommandParserTypes) -> CommandParser:
    if command_parser is CommandParserTypes.PATTERN:
        return PatternCommandParser()
    elif command_parser is CommandParserTypes.AI:
        return AICommandParser()

    raise ValueError(f"Unsupported command parser: {command_parser}")
