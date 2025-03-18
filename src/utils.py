from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to cv2 image
    Args:
        image: PIL opened image
    Returns:
        image: CV2 format NDArray image
    """
    np_image = np.array(image)
    if len(np_image.shape) == 3:
        np_image = np_image[:, :, ::-1].copy()
    return np_image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert cv2 image to PIL format
    Args:
        image: CV2 opened NDArray image
    Returns:
        image: PIL format image
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return pil_image


def get_random_cat_image() -> Image.Image:
    url = "https://api.thecatapi.com/v1/images/search"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    image_url = response.json()[0]["url"]

    image_response = requests.get(image_url, timeout=25)
    image_response.raise_for_status()

    image = Image.open(BytesIO(image_response.content))
    return image
