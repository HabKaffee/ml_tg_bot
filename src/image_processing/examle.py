from io import BytesIO

import requests
from PIL import Image

from src.image_processing.command_parser import PatternCommandParser
from src.image_processing.image_processor import ImageProcessor


def get_random_cat_image() -> Image.Image:
    url = "https://api.thecatapi.com/v1/images/search"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    image_url = response.json()[0]["url"]

    image_response = requests.get(image_url, timeout=25)
    image_response.raise_for_status()

    image = Image.open(BytesIO(image_response.content))
    return image


def main() -> None:
    random_cat_image = get_random_cat_image()
    image_processor = ImageProcessor()
    text_processor = PatternCommandParser()

    random_cat_image.show()

    input_text = "повернуть на 90 градусов"

    commands = text_processor.parse_text(input_text)

    processed_image = image_processor.get_processed_image(image=random_cat_image, command_queue=commands)

    processed_image.show()


if __name__ == "__main__":
    main()
