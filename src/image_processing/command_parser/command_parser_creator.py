from enum import Enum

from src.image_processing.command_parser.ai_command_parser import AICommandParser
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.command_parser.pattern_command_parser import PatternCommandParser


class CommandParserTypes(Enum):
    PATTERN = "pattern"
    AI = "ai"


def get_command_parser(command_parser: CommandParserTypes) -> CommandParser:
    parser_map = {CommandParserTypes.PATTERN: PatternCommandParser, CommandParserTypes.AI: AICommandParser}
    if command_parser not in parser_map:
        raise ValueError(f"Unsupported command parser: {command_parser}")
    return parser_map[command_parser]()
