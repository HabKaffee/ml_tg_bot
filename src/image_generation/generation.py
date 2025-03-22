import torch
from diffusers import DiffusionPipeline
from PIL import Image


class StableDiffusionPipeline:
    def __init__(self, model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self._model_path = model_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = DiffusionPipeline.from_pretrained(self._model_path).to(self._device)

    def generate_image(self, prompt: str) -> Image.Image:
        image: Image.Image = self._pipeline(prompt).images[0]
        return image
