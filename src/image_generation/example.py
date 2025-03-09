from tap import Tap

from src.image_generation.generation import StableDiffusionPipeline


class Arguments(Tap):
    prompt: str = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()
    pipeline = StableDiffusionPipeline()
    image = pipeline.generate_image(args.prompt)
    image.show()


if __name__ == "__main__":
    main()
