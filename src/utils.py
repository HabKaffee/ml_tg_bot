from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageStat
from io import BytesIO
import requests


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


def get_image_size(image: Image.Image) -> Dict[str, int]:
    """
    Get the original size of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        dict: The original size of the image.
    """
    return {"width": image.width, "height": image.height}


def get_average_brightness(image: Image.Image) -> float:
    """
    Get the average brightness of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        float: The average brightness of the image.
    """
    grayscale_image = image.convert("L")
    stat = ImageStat.Stat(grayscale_image)
    return stat.mean[0]


def get_contrast(image: Image.Image) -> float:
    """
    Get the contrast of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        float: The contrast of the image.
    """
    grayscale_image = image.convert("L")
    stat = ImageStat.Stat(grayscale_image)
    return stat.stddev[0]


def get_color_space(image: Image.Image) -> str:
    """
    Get the color space of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        str: The color space of the image.
    """
    return image.mode


def get_saturation(image: Image.Image) -> float:
    """
    Get the saturation of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        float: The saturation of the image.
    """
    hsv_image = image.convert("HSV")
    stat = ImageStat.Stat(hsv_image)
    return stat.mean[1]


def get_level_of_detail(image: Image.Image) -> float:
    """
    Get the level of detail of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        float: The level of detail of the image.
    """
    grayscale_image = image.convert("L")
    laplacian = cv2.Laplacian(np.array(grayscale_image), cv2.CV_64F)
    return float(laplacian.var())


def get_random_cat_image() -> Image.Image:
    url = "https://api.thecatapi.com/v1/images/search"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    image_url = response.json()[0]["url"]

    image_response = requests.get(image_url, timeout=25)
    image_response.raise_for_status()

    image = Image.open(BytesIO(image_response.content))
    return image
