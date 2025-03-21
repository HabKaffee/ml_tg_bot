from tap import Tap

from src.image_processing.command_parser.command_parser import ParserParameters
from src.image_processing.command_parser.command_parser_creator import CommandParserTypes, get_command_parser
from src.image_processing.image_processor import ImageProcessor
from src.utils import get_random_cat_image


class Arguments(Tap):
    command_parser: CommandParserTypes = CommandParserTypes.AI
    input_text: str = "повернуть на 90 градусов и сделать черно белым, подними немного яркость"
    analyze_image: bool = False
    num_few_shot_samples: int = -1


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()

    image_processor = ImageProcessor()
    text_processor = get_command_parser(args.command_parser)

    random_cat_image = get_random_cat_image()
    random_cat_image.show()

    parsing_parameters = ParserParameters(
        num_few_shot_samples=args.num_few_shot_samples,
        analyze_image=args.analyze_image,
        image_to_analyze=random_cat_image,
    )
    commands = text_processor.parse_text(args.input_text, parsing_parameters)

    processed_image = image_processor.get_processed_image(image=random_cat_image, command_queue=commands)

    processed_image.show()


if __name__ == "__main__":
    main()
