from dataclasses import dataclass
from typing import List

from src.image_processing.kernels.kernel_types import KernelTypes


@dataclass
class Command:
    """
    Dataclass for collecting command information
    """

    kernel_type: KernelTypes
    parameters: List[str]
