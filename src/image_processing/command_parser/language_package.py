from abc import ABC, abstractmethod
from enum import Enum
from typing import Collection, Dict, List, Sequence, TypeAlias, Union

from src.image_processing.kernels.kernel_types import KernelTypes


class LanguageType(Enum):
    RU = "ru"
    EN = "en"


FewShotSamples: TypeAlias = List[Dict[str, Sequence[Collection[str]]]]
ImageParameters: TypeAlias = Dict[str, Union[float, str, Dict[str, int]]]


class LanguagePackage(ABC):

    @staticmethod
    @abstractmethod
    def get_patters() -> Dict[KernelTypes, str]: ...

    @staticmethod
    @abstractmethod
    def get_basic_prompt(kernel_keys: List[str], command_keys: List[str]) -> str: ...

    @staticmethod
    @abstractmethod
    def get_few_shot_examples() -> FewShotSamples: ...

    @staticmethod
    @abstractmethod
    def get_image_analysis_prompt(image_parameters: ImageParameters) -> str: ...


class LanguagePackageEN(LanguagePackage):

    @staticmethod
    def get_patters() -> Dict[KernelTypes, str]:
        return {
            KernelTypes.BLUR: r"blur with radius (?P<step>-?\d+)",
            KernelTypes.CONTRAST: r"change contrast by (?P<step>-?\d+)",
            KernelTypes.CROP: r"crop to (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.GRAYSCALE: r"convert to grayscale",
            KernelTypes.INVERT: r"invert colors|invert",
            KernelTypes.RESIZE: r"resize to (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.ROTATE: r"rotate by (?P<angle>-?\d+) degrees",
            KernelTypes.SHARPEN: r"sharpen by (?P<step>-?\d+)",
        }

    @staticmethod
    def get_basic_prompt(kernel_keys: List[str], command_keys: List[str]) -> str:
        return f"""
            You are an intelligent assistant that analyzes text instructions for image processing and converts them into JSON format.

            Input format:
            You are given text that describes how to change the image.

            Output format:
            You should return exactly one JSON object as a list of instructions.
            Each instruction should contain two keys:
            1. `"action"` – the name of the action from the set {kernel_keys}.
            2. `"parameters"` – a dictionary with the parameters needed to perform the action from the set {command_keys}
        """

    @staticmethod
    def get_few_shot_examples() -> FewShotSamples:
        examples = [
            {
                "input": "Turn by 90 degrees and crop to 100 by 150",
                "output": [
                    {"action": "rotate", "parameters": {"angle": 90}},
                    {"action": "crop", "parameters": {"height": 100, "width": 150}},
                ],
            },
            {
                "input": "Increase contrast by 10 and make black and white",
                "output": [
                    {"action": "contrast", "parameters": {"step": 10}},
                    {"action": "grayscale", "parameters": {}},
                ],
            },
        ]
        return examples

    @staticmethod
    def get_image_analysis_prompt(image_parameters: ImageParameters) -> str:
        return f"Take into account the information about the original image: {image_parameters}"


class LanguagePackageRU(LanguagePackage):

    @staticmethod
    def get_patters() -> Dict[KernelTypes, str]:
        return {
            KernelTypes.BLUR: r"размытие с радиусом (?P<step>-?\d+)",
            KernelTypes.CONTRAST: r"изменить контрастность на (?P<step>-?\d+)",
            KernelTypes.CROP: r"обрезать до (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.GRAYSCALE: r"преобразовать в черно-белый",
            KernelTypes.INVERT: r"инвертировать цвета|инвертировать",
            KernelTypes.RESIZE: r"изменить размер до (?P<height>\d+)x(?P<width>\d+)",
            KernelTypes.ROTATE: r"повернуть на (?P<angle>-?\d+) градусов",
            KernelTypes.SHARPEN: r"повысить резкость на (?P<step>-?\d+)",
        }

    @staticmethod
    def get_basic_prompt(kernel_keys: List[str], command_keys: List[str]) -> str:
        return f"""
            Ты — интеллектуальный ассистент, который анализирует текстовые инструкции по обработке изображений и преобразует их в JSON-формат.

            Формат входных данных:
            Тебе дается текст, описывающий, как нужно изменить изображение.

            Формат выходных данных:
            Ты должен вернуть ровно один JSON-объект в виде списка с инструкциями.
            Каждая инструкция должна содержать два ключа:
            1. `"action"` – название действия из набора {kernel_keys}.
            2. `"parameters"` – словарь с параметрами, необходимыми для выполнения действия из набора {command_keys}
        """

    @staticmethod
    def get_few_shot_examples() -> FewShotSamples:
        examples = [
            {
                "input": "Повернуть на 90 градусов и обрезать до 100 на 150",
                "output": [
                    {"action": "rotate", "parameters": {"angle": 90}},
                    {"action": "crop", "parameters": {"height": 100, "width": 150}},
                ],
            },
            {
                "input": "Увеличить контрастность на 10 и сделать черно-белым",
                "output": [
                    {"action": "contrast", "parameters": {"step": 10}},
                    {"action": "grayscale", "parameters": {}},
                ],
            },
        ]
        return examples

    @staticmethod
    def get_image_analysis_prompt(image_parameters: ImageParameters) -> str:
        return f"Прими во внимание информацию об исходном изображении: {image_parameters}"


def get_language_package(language: LanguageType) -> LanguagePackage:
    language_map = {LanguageType.EN: LanguagePackageEN, LanguageType.RU: LanguagePackageRU}
    if language not in language_map:
        raise ValueError(f"Unsupported language: {language}")
    return language_map[language]()
