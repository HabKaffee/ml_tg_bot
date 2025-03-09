import PIL
from diffusers import DiffusionPipeline

class StableDiffusionPipeline:
    def __init__(self, model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self._model_name = model_name
        self._pipeline = DiffusionPipeline.from_pretrained(self._model_name)
    
    def generate_image(self, prompt: str) -> PIL.Image.Image:
        image = self._pipeline(prompt).images[0]
        return image
