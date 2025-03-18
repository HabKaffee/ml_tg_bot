from pathlib import Path

from src.sticker_generator.sticker_generator import StickerGenerator
from src.utils import get_random_cat_image

basePath = "data/processed_stickers"
Path(basePath).mkdir(parents=True, exist_ok=True)


def test_sticker_generator() -> None:
    sticker_gen = StickerGenerator()
    input_image = get_random_cat_image().convert("RGB")

    input_path = f"{basePath}/test_input.png"
    input_image.save(input_path)

    output_path = f"{basePath}/result.png"
    result_image = sticker_gen.generate_sticker(input_image)

    if result_image:
        result_image.save(output_path)
        print(f"Result sticker saved in: {output_path}")
    else:
        print("Sticker generation failed")


if __name__ == "__main__":
    test_sticker_generator()
