import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import pipeline

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


def process_json_text(input_text: str) -> str:
    return input_text.replace("\n", "").replace("'", '"')


import cv2
import numpy as np
from PIL import Image, ImageStat


class AIPrompter:
    def __init__(self) -> None:
        self._basic_prompt = self._prepare_basic_prompt()
        self._path_to_few_shot_examples = Path("src/image_processing/command_parser/few_shot_examples.json")
        self._few_shot_prompt = self._prepare_few_shot_prompt()

    def _prepare_few_shot_prompt(self) -> str:
        with self._path_to_few_shot_examples.open(mode="r", encoding="utf-8") as file:
            few_shot_examples = json.load(file)

        few_shot_prompt = "Examples:\n"
        for example in few_shot_examples:
            input_text = example["input"]
            expected_json = example["output"]
            few_shot_prompt += f"Input: {input_text}. Result: {expected_json}.\n"
        return few_shot_prompt

    def _prepare_basic_prompt(self) -> str:
        kernel_keys = KernelTypes.get_keys()
        command_keys = CommandParameters().get_keys()

        prompt = f"""
        Ты — интеллектуальный ассистент, который анализирует текстовые инструкции по обработке изображений и преобразует их в JSON-формат.

        Формат входных данных:
        Тебе дается текст, описывающий, как нужно изменить изображение.

        Формат выходных данных:
        Ты должен вернуть ровно один JSON-объект в виде списка с инструкциями.
        Каждая инструкция должна содержать два ключа:
        1. `"action"` – название действия из набора {kernel_keys}.
        2. `"parameters"` – словарь с параметрами, необходимыми для выполнения действия из набора {command_keys}
        """
        return prompt

    def prepare_prompt(self) -> str:
        prompt = f"""
        {self._basic_prompt}
        ---
        {self._few_shot_prompt}
        """
        return prompt


def get_image_size(image: Image.Image) -> tuple:
    """
    Get the original size of the image.

    Args:
        image (Image.Image): The image to be analyzed.

    Returns:
        tuple: The width and height of the image.
    """
    return image.size


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


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def __init__(self) -> None:
        super().__init__()
        self._prompter = AIPrompter()
        self._json_pattern = re.compile(r".*(?P<json>\[.*\]).*")

        self._image_parameters: Dict[str, Any] = {}

        self._model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._pipeline = pipeline(
            "text-generation", model=self._model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def _get_llm_output(self, input_text: str) -> List[Dict[str, Union[str, Dict]]]:
        prompt = self._prompter.prepare_prompt()
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": input_text},
        ]
        if self._image_parameters:
            messages.append(
                {"role": "user", "content": f"Вот информация об исходном изображении {self._image_parameters}"}
            )

        prompt = self._pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self._pipeline(prompt)

        processed_text = process_json_text(outputs[0]["generated_text"])
        processed_text_match = self._json_pattern.match(processed_text)

        if processed_text_match:
            return list(json.loads(processed_text_match.group("json")))
        raise ValueError(f"Failed to parse text: {input_text}")

    def parse_text(self, text: str) -> List[Command]:
        json_command = self._get_llm_output(text)

        commands = []
        for command in json_command:
            kernel_type = KernelTypes(command["action"])

            parameters = command["parameters"]
            if not isinstance(parameters, dict):
                raise ValueError(f"Failed to parse parameters: {parameters}")

            command_parameters = CommandParameters(**parameters)
            commands.append(Command(kernel_type, command_parameters))
        return commands

    def analyze_image(self, image: Image.Image) -> None:
        self._image_parameters = {
            "original_size": get_image_size(image),
            "average_brightness": get_average_brightness(image),
            "contrast": get_contrast(image),
            "color_space": get_color_space(image),
            "saturation": get_saturation(image),
            "level_of_detail": get_level_of_detail(image),
        }
