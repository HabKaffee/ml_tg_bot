from enum import Enum


class KernelTypes(Enum):
    """
    The list of supported commands
    """

    BLUR = "blur"
    CONTRAST = "contrast"
    CROP = "crop"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    RESIZE = "resize"
    ROTATE = "rotate"
    SHARPEN = "sharpen"
