from dataclasses import dataclass
from typing import List

from src.image_processing.kernels.kernel_types import KernelTypes


@dataclass
class CommandParameters:
    """
    Dataclass for collecting command parameters
    """

    step: str = "0"
    height: str = "0"
    width: str = "0"
    angle: str = "0"

    @classmethod
    def get_keys(cls) -> List[str]:
        return list(cls.__annotations__.keys())


@dataclass
class Command:
    """
    Dataclass for collecting command information
    """

    kernel_type: KernelTypes
    parameters: CommandParameters
