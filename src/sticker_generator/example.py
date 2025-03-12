import os
from src.utils import get_random_cat_image
from src.sticker_generator import StickerGenerator

os.makedirs("tmp", exist_ok=True)


def test_sticker_generator():
    sticker_gen = StickerGenerator()
    input_image = get_random_cat_image().convert("RGB")

    input_path = "tmp/test_input.png"
    input_image.save(input_path)

    output_path = "tmp/result.png"
    result_image = sticker_gen.generate_sticker(input_image)

    if result_image:
        result_image.save(output_path)
        print(f"Result sticker saved in: {output_path}")
    else:
        print("Sticker generation failed")


if __name__ == "__main__":
    test_sticker_generator()
