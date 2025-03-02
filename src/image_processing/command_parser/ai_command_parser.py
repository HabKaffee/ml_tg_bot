import json
import re
from typing import List

import torch
from accelerate.test_utils.testing import get_backend
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.image_processing.command import Command, CommandParameters
from src.image_processing.command_parser.command_parser import CommandParser
from src.image_processing.kernels.kernel_types import KernelTypes


class AIPrompter:
    def __init__(self):
        self._basic_prompt = self._prepare_basic_prompt()
        self._few_shot_prompt = self._prepare_few_shot_prompt()

    def _prepare_few_shot_prompt(self) -> str:
        few_shot_examples = [
            (
                "Повернуть на 90 градусов и обрезать до 100 на 150",
                '[{"action":"rotate","parameters":{"angle":90}},{"action":"crop","parameters":{"height":100,"width":150}}]',
            ),
            (
                "Увеличить контрастность на 10 и сделать черно-белым",
                '[{"action":"contrast","parameters":{"step":10}},{"action":"grayscale","parameters":{None:None}}]',
            ),
        ]

        few_shot_prompt = "Examples:\n"
        for example in few_shot_examples:
            input_text, expected_json = example
            few_shot_prompt += f'Input: "{input_text}". Result: "{expected_json}".\n'
        return few_shot_prompt

    def _prepare_basic_prompt(self) -> str:
        kernel_keys = KernelTypes.get_keys()
        command_keys = CommandParameters().get_keys()

        prompt = f"""
        Ты — интеллектуальный ассистент, который анализирует текстовые инструкции по обработке изображений и преобразует их в JSON-формат.

        Формат входных данных:
        Тебе дается текст, описывающий, как нужно изменить изображение.

        Формат выходных данных:
        Ты должен вернуть JSON-объект с инструкциями.
        JSON должен содержать два ключа:
        1. `"action"` – название действия из набора {kernel_keys}.
        2. `"parameters"` – словарь с параметрами, необходимыми для выполнения действия из набора {command_keys}
        """
        return prompt

    def prepare_prompt(self, input_text: str, expected_json_key: str) -> str:
        prompt = f"""
        {self._basic_prompt}
        ---
        {self._few_shot_prompt}
        ---
        Входной текст:
        "{input_text}"

        {expected_json_key}
        """
        return prompt


class AICommandParser(CommandParser):
    """
    Command parser based on LLM
    """

    def __init__(self):
        super().__init__()
        self._model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self._device, _, _ = get_backend()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=torch.float16, device_map="auto"
        ).to(self._device)

        self._prompter = AIPrompter()
        self._json_pattern = re.compile(r".*(?P<json>\[.*\]).*")

    def _get_llm_output(self, input_text: str) -> str:
        expected_json_key = "##########"
        prompt = self._prompter.prepare_prompt(input_text, expected_json_key)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs)
        parsed_text = self._tokenizer.decode(output[0], skip_special_tokens=True)

        clipped_parsed_text = parsed_text.split(expected_json_key)[1].strip().replace("\n", "").replace(" ", "")

        parsed_text_match = self._json_pattern.match(clipped_parsed_text)
        if parsed_text_match:
            return json.loads(parsed_text_match.group("json"))
        raise ValueError(f"Failed to parse text: {input_text}")

    def parse_text(self, text: str) -> List[Command]:

        json_command = self._get_llm_output(text)

        commands = []
        for command in json_command:
            action = command["action"]
            parameters = command["parameters"]

            kernel_type = KernelTypes(action)
            parameters = CommandParameters(**parameters)

            commands.append(Command(kernel_type, parameters))

        return commands
