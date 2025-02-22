from src.image_processing.image_processing import ImageProcessorByText
from src.image_processing.random_image import get_random_cat_image


def main() -> None:
    random_cat_image = get_random_cat_image()
    processor = ImageProcessorByText()

    random_cat_image.show()

    command = "повернуть на 90 градусов"

    processed_image = processor.get_processed_image(random_cat_image, command)

    processed_image.show()


if __name__ == "__main__":
    main()
