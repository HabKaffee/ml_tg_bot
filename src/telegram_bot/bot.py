from logging import Logger
from pathlib import Path

from PIL import Image
from telegram import InlineKeyboardMarkup, Update, Voice
from telegram.ext import ContextTypes, ConversationHandler

from src.audio2text.speech_recognition import SpeechRecognition
from src.image_generation.generation import StableDiffusionPipeline
from src.image_processing.command_parser.command_parser import ParserParameters
from src.image_processing.command_parser.command_parser_creator import CommandParserTypes, get_command_parser
from src.image_processing.command_parser.language_package import LanguageType
from src.image_processing.image_processor import ImageProcessor
from src.sticker_generator.sticker_generator import StickerGenerator

from .utils import (
    BOT_STATES,
    KEYBOARD,
    PROMPT_IF_CONTINUE_EDIT,
    PROMPT_IF_CONTINUE_GENERATE,
    PROMPT_IF_CONTINUE_STICKER,
    PROMPT_IF_CONTINUE_TRANSCRIBE,
)


class TelegramBot:
    def __init__(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        audio_processor: SpeechRecognition,
        image_processor: ImageProcessor,
        sticker_processor: StickerGenerator,
        logger: Logger,
        data_folder: Path = Path("data/"),
        num_few_shot_samples: int = -1,
        analyze_image: bool = False,
    ) -> None:
        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.sticker_processor = sticker_processor
        self.image_generator = StableDiffusionPipeline()
        self.command_parser = get_command_parser(CommandParserTypes.PATTERN)(LanguageType.EN)
        self.parsing_parameters = ParserParameters(
            num_few_shot_samples=num_few_shot_samples,
            analyze_image=analyze_image,
            image_to_analyze=None,
        )
        self.logger = logger
        self.data_folder = data_folder

    async def start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Starts the conversation
        """
        self.logger.info("EVENT: Start")
        menu_keyboard = InlineKeyboardMarkup(KEYBOARD)

        if update.message:
            self.logger.info("Rendered keyboard in start")
            await update.message.reply_text(
                "Hi! Welcome to the bot. Choose the command to continue",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in start")

        return BOT_STATES.ACTION_SELECTION

    async def photo_to_sticker_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a photo."""
        self.logger.info("EVENT: waiting for photo to prepare sticker")
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text(  # type: ignore[union-attr]
                "Please send a photo to prepare sticker. "
                "Use photo with an object placed in the center. Send photo 1 by 1"
            )
        else:
            self.logger.error("Exception in photo_to_sticker_prompt")
        return BOT_STATES.PHOTO_STICKER

    async def edit_photo_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a photo."""
        self.logger.info("EVENT: waiting for photo to edit")
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send a photo to edit.")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in edit_photo_prompt")
        return BOT_STATES.PHOTO_EDIT

    async def audio_to_text_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a audio."""
        self.logger.info("EVENT: waiting for audio to perform audio2text")
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send audio to transcribe.")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in audio_to_text_prompt")
        return BOT_STATES.AUDIO

    async def generate_image_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a prompt."""
        self.logger.info("EVENT: waiting for prompt to generate image")
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send a prompt to generate image")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in generate_image_prompt")
        return BOT_STATES.GENERATE

    async def photo_to_sticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to gathed needed data and call sticker pack creation model
        """
        self.logger.info("EVENT: photo_to_sticker")
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)

        if not update.effective_message.photo:
            await update.effective_message.reply_text("Please provide photo first.")
            return await self.restart(update, context)

        await update.effective_message.reply_text("Image is being processed. Please wait...")

        photo = await update.effective_message.photo[-1].get_file()

        input_image_path = None
        output_image_path = None

        if update.effective_user:
            input_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_sticker.png")
            output_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_sticker_generated.png")

        if not input_image_path or not output_image_path:
            self.logger.error("Input and/or output image path is not set!")
            await update.effective_message.reply_text("Internal error occured. Please try again.")
            return await self.photo_to_sticker_continue(update, context)

        await photo.download_to_drive(input_image_path)

        if not input_image_path.exists():
            await update.effective_message.reply_text("There is a problem with provided photo. Please, resend it.")
            self.logger.error("Error occured during sticker generation : No data is provided.")
            return await self.photo_to_sticker_continue(update, context)

        input_image = Image.open(input_image_path).convert("RGB")
        result_image = self.sticker_processor.generate_sticker(input_image)

        if not result_image:
            await update.effective_message.reply_text("An error occured during sticker generation")
            self.logger.error("Error occured during sticker generation : No sticker is generated.")
            return await self.restart(update, context)

        result_image.save(output_image_path)
        with open(output_image_path, "rb") as photo_file:
            await update.effective_message.reply_document(photo_file, filename="prepared_sticker.png")
            self.logger.info("Sticker generated successfully")

        return await self.photo_to_sticker_continue(update, context)

    async def audio_to_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to gathed needed data and call audio to text conversion model
        """
        self.logger.info("EVENT: audio2text")
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)

        if not update.effective_user:
            self.logger.warning("Update.effective_user is empty")
            return await self.restart(update, context)

        input_audio_path = Path(f"{self.data_folder}/{update.effective_user.id}_audio_in.ogg")

        if update.effective_message and not update.effective_message.voice:
            await update.effective_message.reply_text("No audio is provided. Please provide one")
            return await self.audio_to_text_continue(update, context)

        await update.effective_message.reply_text("Transcription in progress...")
        if update.effective_message and isinstance(update.effective_message.voice, Voice):
            audio_file = await update.effective_message.voice.get_file()
        else:
            self.logger.error("No voice is provided.")
            return await self.audio_to_text_continue(update, context)

        await audio_file.download_to_drive(input_audio_path)
        generated_transcription = self.audio_processor.gen_transcription(input_audio_path)

        self.logger.info("Audio transcribed successfully")
        await update.effective_message.reply_text(f"Here is the transcribed message:\n\n{generated_transcription}")

        return await self.audio_to_text_continue(update, context)

    async def edit_photo(  # pylint: disable=too-many-return-statements
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> BOT_STATES:
        """
        Handler to gathed needed data and call picture edit model
        """
        self.logger.info("EVENT: edit_photo")
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)

        if not update.effective_message.photo:
            await update.effective_message.reply_text("Please provide photo first.")
            return await self.restart(update, context)

        photo = await update.effective_message.photo[-1].get_file()
        description = None

        if update.effective_message.caption:
            description = update.effective_message.caption

        if not description:
            await update.effective_message.reply_text(
                "No caption is provided! Please, ensure you provide desired changes in caption to photo"
            )
            return await self.edit_photo_continue(update, context)

        input_image_path = None
        output_image_path = None

        if update.effective_user:
            input_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_edit.png")
            output_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_edited_photo.png")

        if not input_image_path or not output_image_path:
            self.logger.error("Input and/or output image path is not set!")
            await update.effective_message.reply_text("Internal error occured. Please try again.")
            return await self.edit_photo_continue(update, context)

        await photo.download_to_drive(input_image_path)

        if not input_image_path.exists():
            await update.effective_message.reply_text("There is a problem with provided photo. Please, resend it.")
            self.logger.error("Error occured during image changing : No data is provided.")
            return await self.photo_to_sticker_continue(update, context)

        input_image = Image.open(input_image_path).convert("RGB")

        if self.parsing_parameters.analyze_image:
            self.parsing_parameters.image_to_analyze = input_image

        commands = self.command_parser.parse_text(description, self.parsing_parameters)

        self.logger.info("%s -> %s", description, commands)
        await update.effective_message.reply_text(f"Processing image with command: '{description}'")

        edited_photo = self.image_processor.get_processed_image(image=input_image, command_queue=commands)

        if not edited_photo:
            await update.effective_message.reply_text("An error occured during photo editing")
            self.logger.error("Error occured during photo editing : Photo not edited.")
            return await self.edit_photo_continue(update, context)

        edited_photo.save(output_image_path)
        with open(output_image_path, "rb") as photo_file:
            await update.effective_message.reply_document(photo_file, filename="edited_photo.png")
            self.logger.info("photo edited successfully")

        return await self.edit_photo_continue(update, context)

    async def generate_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to gathed needed data and call audio to text conversion model
        """
        self.logger.info("EVENT: generate_image")
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)

        if not update.effective_user:
            self.logger.warning("Update.effective_user is empty")
            return await self.restart(update, context)

        output_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_generated_image.png")

        if update.effective_message and not update.effective_message.text:
            await update.effective_message.reply_text("No prompt is provided. Please provide one")
            return await self.generate_image_continue(update, context)

        await update.effective_message.reply_text("Generation in progress... It may take a while. You'll get notified")
        prompt = update.effective_message.text

        if not prompt:
            self.logger.error("No prompt provided")
            return await self.generate_image_continue(update, context)

        generated_image = self.image_generator.generate_image(prompt)
        if not generated_image:
            self.logger.error("No image generated.")
            await update.effective_message.reply_text("An error occured during image generation. Try again")

        generated_image.save(output_image_path)

        with open(output_image_path, "rb") as photo_file:
            await update.effective_message.reply_document(photo_file, filename="generated.png")
            self.logger.info("image is generated successfully")

        return await self.generate_image_continue(update, context)

    async def photo_to_sticker_continue(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to process more photos to sticker or return to menu
        """
        self.logger.info("EVENT: photo_to_sticker contunue")
        menu_keyboard = InlineKeyboardMarkup(PROMPT_IF_CONTINUE_STICKER)

        if update.message:
            self.logger.info("Rendered keyboard in continue_sticker")
            await update.message.reply_text(
                "Do you want to prepare more photos?",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in photo_to_sticker_continue")

        return BOT_STATES.ACTION_SELECTION

    async def edit_photo_continue(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to process more photos or return to menu
        """
        self.logger.info("EVENT: edit_photo contunue")
        menu_keyboard = InlineKeyboardMarkup(PROMPT_IF_CONTINUE_EDIT)

        if update.message:
            self.logger.info("Rendered keyboard in continue_edit")
            await update.message.reply_text(
                "Do you want to prepare more photos?",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in edit_photo_continue")

        return BOT_STATES.ACTION_SELECTION

    async def audio_to_text_continue(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to process more audios or return to menu
        """
        self.logger.info("EVENT: audio2text contunue")
        menu_keyboard = InlineKeyboardMarkup(PROMPT_IF_CONTINUE_TRANSCRIBE)

        if update.message:
            self.logger.info("Rendered keyboard in continue_audio")
            await update.message.reply_text(
                "Do you want to prepare more audios?",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in audio2text_continue")

        return BOT_STATES.ACTION_SELECTION

    async def generate_image_continue(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to generate more images or return to menu
        """
        self.logger.info("EVENT: generate_image contunue")
        menu_keyboard = InlineKeyboardMarkup(PROMPT_IF_CONTINUE_GENERATE)

        if update.message:
            self.logger.info("Rendered keyboard in continue_")
            await update.message.reply_text(
                "Do you want to generate more images?",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in generate_image_continue")

        return BOT_STATES.ACTION_SELECTION

    async def cancel(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancels the process and returns to the main menu."""
        self.logger.info("EVENT: cancel")
        if update.effective_message:
            await update.effective_message.reply_text("Okay, process canceled. Type /start to restart.")
        else:
            self.logger.error("Exception in cancel")
        return ConversationHandler.END

    async def restart(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        self.logger.info("EVENT: Restart")
        menu_keyboard = InlineKeyboardMarkup(KEYBOARD)
        if update.effective_message:
            self.logger.info("Rendered keyboard in restart")
            await update.effective_message.reply_text(
                "Choose the command to continue",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Can't render keyboard")
        return BOT_STATES.ACTION_SELECTION
