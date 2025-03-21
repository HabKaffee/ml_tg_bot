import json
import logging
import re
from typing import Dict, List, Union

import torch
from PIL import Image
from transformers import pipeline

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser, ParserParameters
from src.image_processing.command_parser.language_package import FewShotSamples, ImageParameters, LanguageType
from src.image_processing.kernels.kernel_types import KernelTypes
from src.utils import (
    get_average_brightness,
    get_color_space,
    get_contrast,
    get_image_size,
    get_level_of_detail,
    get_saturation,
)


def process_json_text(input_text: str) -> str:
    return input_text.replace("\n", "").replace("'", '"')


class AIPrompter:
    def __init__(self, language: LanguageType = LanguageType.EN) -> None:
        self._basic_prompt = self._prepare_basic_prompt()
        self._language = language
        self._few_shot_samples = self._language_package.get_few_shot_examples()

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
        prompt = self._language_package.get_basic_prompt(kernel_keys, command_keys)
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
            image_prompt = self._language_package.get_image_analysis_prompt(self._image_parameters)

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
