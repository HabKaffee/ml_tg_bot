from tap import Tap

from src.image_generation.generation import AspectRatio, StableDiffusionPipeline, Style


class Arguments(Tap):
    prompt: str = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    aspect_ratio: AspectRatio = "1:1"
    style: Style = "default"
    
    def post_process(self):
        self.aspect_ratio = AspectRatio(self.aspect_ratio)
        self.style = Style(self.style)

def generate(args: Arguments, pipeline: StableDiffusionPipeline) -> None:
    image = pipeline.generate_image(args.prompt, args.aspect_ratio, args.style)
    image.show()

if __name__ == "__main__":
    pipeline = StableDiffusionPipeline()
    ARGS = Arguments(underscores_to_dashes=True).parse_args()
    print(ARGS)
    generate(ARGS, pipeline)
