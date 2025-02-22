from typing import Dict

from src.image_processing.kernels.blur import BlurImage
from src.image_processing.kernels.contrast import ContrastImage
from src.image_processing.kernels.crop import CropImage
from src.image_processing.kernels.grayscale import GrayscaleImage
from src.image_processing.kernels.invert import InvertImage
from src.image_processing.kernels.kernel import Kernel
from src.image_processing.kernels.kernel_types import KernelTypes
from src.image_processing.kernels.resize import ResizeImage
from src.image_processing.kernels.rotate import RotateImage
from src.image_processing.kernels.sharpen import SharpenImage


def get_kernel_map() -> Dict[KernelTypes, type[Kernel]]:
    """
    Get the map of kernel types to kernel classes
    Returns:
        Dict[KernelTypes, Kernel]: the map of kernel types to kernel classes
    """
    return {
        KernelTypes.RESIZE: ResizeImage,
        KernelTypes.ROTATE: RotateImage,
        KernelTypes.CONTRAST: ContrastImage,
        KernelTypes.CROP: CropImage,
        KernelTypes.GRAYSCALE: GrayscaleImage,
        KernelTypes.INVERT: InvertImage,
        KernelTypes.BLUR: BlurImage,
        KernelTypes.SHARPEN: SharpenImage,
    }
