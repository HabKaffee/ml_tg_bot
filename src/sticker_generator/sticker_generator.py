from typing import Optional

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor  # type: ignore
from PIL import Image

from src.utils import pil_to_cv2


class StickerGenerator:
    """
    Generates stickers by segmenting the main object in an image using model and postprocess afterwards.
    """

    def __init__(self, model_path: str = "models/sam_vit_h_4b8939.pth", model_type: str = "vit_h") -> None:
        """
        Initialize the StickerGenerator with model.

        Args:
            model_path: Path to the SAM model checkpoint. Defaults to "models/sam_vit_h_4b8939.pth".
            model_type: Type of SAM model architecture. Defaults to "vit_h".
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        self._model = SamPredictor(sam)

    def generate_sticker(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Generate a sticker by segmenting the central object in the image.

        Args:
            image: Input PIL image to process.

        Returns:
            Processed PIL image with transparency, cropped and resized to 512x512.
            Returns `None` if segmentation fails.
        """
        cv2_image = pil_to_cv2(image)
        self._model.set_image(cv2_image)

        H, W = cv2_image.shape[:2]
        point_coords = np.array([[W // 2, H // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)

        masks, scores, _ = self._model.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True
        )

        index = np.argmax(scores)

        if len(masks) == 0:
            return None

        mask = masks[index].astype(np.uint8)
        mask = self._postprocess_mask(mask)

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA)
        result = self._apply_mask(cv2_image, mask)

        coords = cv2.findNonZero(mask)
        if coords is None or coords.size == 0:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped = result.crop((x, y, x + w, y + h))

        final = self._resize_to_sticker(cropped)

        return final

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the raw segmentation mask using thresholding and morphological operations.

        Args:
            mask: Raw segmentation mask (numpy array).

        Returns:
            Processed binary mask.
        """
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """
        Apply the mask to the image to create transparency.

        Args:
            image: Input image in RGBA format (numpy array).
            mask: Binary mask (numpy array).

        Returns:
            PIL image with alpha channel applied.
        """
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255).convert("L")
        image_pil.putalpha(mask_pil)
        return image_pil

    def _resize_to_sticker(self, image: Image.Image) -> Image.Image:
        """
        Resize the image to 512x512 while preserving aspect ratio and centering.

        Args:
            image: Cropped PIL image.

        Returns:
            Centered 512x512 PIL image with transparent padding.
        """
        max_size = 512
        w, h = image.size
        aspect_ratio = w / h

        if w > h:
            new_w = max_size
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = max_size
            new_w = int(aspect_ratio * max_size)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        final = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
        final.paste(image, ((max_size - new_w) // 2, (max_size - new_h) // 2))
        return final
