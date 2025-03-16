from diffusers import DiffusionPipeline
from PIL import Image
from enum import Enum
import torch


class AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"


class Style(Enum):
    DEFAULT = "default"
    DETAILED_PHOTO = "detailed photo"
    DIGITAL_PAINTING = "digital painting"
    ANIME = "anime"


RESOLUTION_MAP = {
    AspectRatio.RATIO_16_9: (1024, 576),
    AspectRatio.RATIO_9_16: (576, 1024),
    AspectRatio.RATIO_1_1: (768, 768),
}

STYLE_PROMPTS = {
    Style.DEFAULT: "",
    Style.DETAILED_PHOTO: "highly detailed professional photograph, ultra-realistic, 8K",
    Style.DIGITAL_PAINTING: "digital painting, concept art, highly detailed, vibrant colors",
    Style.ANIME: "anime style, highly detailed, colorful, cel shading",
}


class StableDiffusionPipeline:
    def __init__(self, model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self._model_path = model_path
        self._pipeline = DiffusionPipeline.from_pretrained(
            self._model_path, 
            torch_dtype=torch.float16,   
            safety_checker=None
        ).to("cuda")


    def generate_image(
        self,
        prompt: str,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1,
        style: Style = Style.DEFAULT,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        
        width, height = RESOLUTION_MAP[aspect_ratio]
        full_prompt = f"{prompt}, {STYLE_PROMPTS[style]}".strip(", ")

        image = self._pipeline(
            prompt=full_prompt, 
            width=width, 
            height=height, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        ).images[0]
        return image
