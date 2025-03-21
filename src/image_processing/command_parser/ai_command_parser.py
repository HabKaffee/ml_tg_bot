import json
import logging
import re
from pathlib import Path
from typing import Dict, List, TypeAlias, Union

import torch
from PIL import Image
from transformers import pipeline

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser, ParserParameters
from src.image_processing.command_parser.language import LanguageType
from src.image_processing.kernels.kernel_types import KernelTypes
from src.utils import (
    get_average_brightness,
    get_color_space,
    get_contrast,
    get_image_size,
    get_level_of_detail,
    get_saturation,
)

FewShotSamples: TypeAlias = List[Dict[str, Union[List[Dict[str, Union[str, int]]]]]]
ImageParameters: TypeAlias = Dict[str, Union[float, str, Dict[str, int]]]


def process_json_text(input_text: str) -> str:
    return input_text.replace("\n", "").replace("'", '"')


class AIPrompter:
    def __init__(self, language: LanguageType = LanguageType.EN) -> None:
        self._basic_prompt = self._prepare_basic_prompt()
        self._language = language
        self._path_to_few_shot_examples = Path(f"src/image_processing/command_parser/few_shot_examples_{self._language }.json")
        self._few_shot_samples = self._load_few_shot_prompt(self._path_to_few_shot_examples)

    @staticmethod
    def _load_few_shot_prompt(path_to_few_shot_examples: Path) -> FewShotSamples:
        few_shot_samples = []
        if not path_to_few_shot_examples.exists():
            logging.warning("There is no file %s with Few-Shot examples", path_to_few_shot_examples)
            return few_shot_samples

        try:
            with path_to_few_shot_examples.open(mode="r", encoding="utf-8") as file:
                few_shot_samples = json.load(file)
        except json.decoder.JSONDecodeError as exc:
            logging.warning("Loading file with Few-Shot examples raised %s exception", exc)
        return few_shot_samples

    @staticmethod
    def _prepare_few_shot_prompt(few_shot_samples: FewShotSamples, num_few_shot_samples: int = -1) -> str:
        _num_samples = min(num_few_shot_samples, len(few_shot_samples))

        few_shot_prompt = "Examples:\n"
        for example in few_shot_samples[:_num_samples]:
            input_text = example["input"]
            expected_json = example["output"]
            few_shot_prompt += f"Input: {input_text}. Result: {expected_json}.\n"
        return few_shot_prompt

    def _prepare_basic_prompt(self) -> str:
        kernel_keys = KernelTypes.get_keys()
        command_keys = CommandParameters().get_keys()

        if self._language is LanguageType.EN:
            prompt = f"""
            You are an intelligent assistant that analyzes text instructions for image processing and converts them into JSON format.

            Input format:
            You are given text that describes how to change the image.

            Output format:
            You should return exactly one JSON object as a list of instructions.
            Each instruction should contain two keys:
            1. `"action"` – the name of the action from the set {kernel_keys}.
            2. `"parameters"` – a dictionary with the parameters needed to perform the action from the set {command_keys}
            """
        elif self._language is LanguageType.RU:
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
        else:
            raise ValueError(f"Unsupported language: {self._language})")
        return prompt

    def prepare_prompt(self, num_few_shot_samples: int = -1) -> str:
        prompt = self._basic_prompt

        if num_few_shot_samples != 0:
            prompt += "\n" + self._prepare_few_shot_prompt(self._few_shot_samples, num_few_shot_samples)
        return prompt


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def __init__(self, language: LanguageType = LanguageType.EN) -> None:
        super().__init__(language)
        self._prompter = AIPrompter(language)
        self._json_pattern = re.compile(r".*(?P<json>\[.*\]).*")

        self._image_parameters: ImageParameters = {}

        self._model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._pipeline = pipeline(
            "text-generation", model=self._model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def _get_llm_output(self, input_text: str, num_few_shot_samples: int = -1) -> List[Dict[str, Union[str, Dict]]]:
        messages = []

        messages.append({"role": "system", "content": self._prompter.prepare_prompt(num_few_shot_samples)})
        if self._image_parameters:
            if self._language is LanguageType.EN:
                image_prompt = f"Take into account the information about the original image: {self._image_parameters}"
            elif self._language is LanguageType.RU:
                image_prompt = f"Прими во внимание информацию об исходном изображении: {self._image_parameters}"
            else:
                raise ValueError(f"Unsupported language: {self._language}")

            messages.append({"role": "system", "content": image_prompt})

        messages.append({"role": "user", "content": input_text})

        prompt = self._pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self._pipeline(prompt)

        processed_text = process_json_text(outputs[0]["generated_text"])
        logging.info("Received from LLM: %s", processed_text)

        processed_text_match = self._json_pattern.match(processed_text)

        if processed_text_match:
            return list(json.loads(processed_text_match.group("json")))
        raise ValueError(f"Failed to parse text: {input_text}")

    def parse_text(self, text: str, parser_parameters: ParserParameters) -> List[Command]:
        if parser_parameters.analyze_image and parser_parameters.image_to_analyze is not None:
            self._image_parameters = self._analyze_image(parser_parameters.image_to_analyze)
            logging.info("Original image parameters: %s", self._image_parameters)

        json_command = self._get_llm_output(text, parser_parameters.num_few_shot_samples)

        commands = []
        for command in json_command:
            kernel_type = KernelTypes(command["action"])

            parameters = command["parameters"]
            if not isinstance(parameters, dict):
                raise ValueError(f"Failed to parse parameters: {parameters}")

            command_parameters = CommandParameters(**parameters)
            commands.append(Command(kernel_type, command_parameters))
        return commands

    @staticmethod
    def _analyze_image(image: Image.Image) -> ImageParameters:
        image_parameters = {
            "original_size": get_image_size(image),
            "average_brightness": get_average_brightness(image),
            "contrast": get_contrast(image),
            "color_space": get_color_space(image),
            "saturation": get_saturation(image),
            "level_of_detail": get_level_of_detail(image),
        }
        return image_parameters
