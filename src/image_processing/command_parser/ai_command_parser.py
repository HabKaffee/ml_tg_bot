import json
import re
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import pipeline

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


def process_json_text(input_text: str) -> str:
    return input_text.replace("\n", "").replace("'", '"')


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


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def __init__(self) -> None:
        super().__init__()
        self._prompter = AIPrompter()
        self._json_pattern = re.compile(r".*(?P<json>\[.*\]).*")

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
