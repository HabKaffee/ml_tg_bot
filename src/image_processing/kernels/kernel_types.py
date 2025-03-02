from enum import Enum
from typing import List


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

    def get_keys() -> List[str]:
        return [key.value for key in KernelTypes]
