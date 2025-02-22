from io import BytesIO

import requests
from PIL import Image


def get_random_cat_image() -> Image.Image:
    url = "https://api.thecatapi.com/v1/images/search"
    response = requests.get(url)
    response.raise_for_status()
    image_url = response.json()[0]["url"]

    image_response = requests.get(image_url)
    image_response.raise_for_status()

    image = Image.open(BytesIO(image_response.content))
    return image
