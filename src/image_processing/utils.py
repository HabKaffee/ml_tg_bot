import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL image to cv2 image
    Args:
        image: PIL opened image
    Returns:
        image: CV2 format image
    """
    image = np.array(image)
    if len(image.shape) == 3:
        image = image[:, :, ::-1].copy()
    return image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert cv2 image to PIL format
    Args:
        image: CV2 opened image
    Returns:
        image: PIL format image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image
