from logging import Logger
from pathlib import Path
from typing import Any

from PIL import Image
from telegram import InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from src.audio2text.speech_recognition import SpeechRecognition
from src.image_processing.image_processor import ImageProcessor
from src.sticker_generator.sticker_generator import StickerGenerator

from .utils import BOT_STATES, KEYBOARD, PROMPT_IF_CONTINUE_EDIT, PROMPT_IF_CONTINUE_STICKER, PROMPT_IF_CONTINUE_TRANSCRIBE


class TelegramBot:
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        audio_processor: SpeechRecognition,
        image_processor: ImageProcessor,
        sticker_processor: StickerGenerator,
        logger: Logger,
        data_folder: Path = Path("data/"),
    ) -> None:
        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.sticker_processor = sticker_processor
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
            await query.message.reply_text("Please send a photo to prepare sticker. For correct sticker, please, use photo with an object placed in the center")  # type: ignore[union-attr]
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
        """Handles the button click and asks for a photo."""
        self.logger.info("EVENT: waiting for audio to perform audio2text")
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send audio to transcribe.")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in audio_to_text_prompt")
        return BOT_STATES.AUDIO

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
        if update.effective_user:
            path = f"{self.data_folder}/{update.effective_user.id}_sticker.png"
        else:
            path = f"{self.data_folder}/unknown_user_id_sticker.png"
        await photo.download_to_drive(path)

        input_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_sticker.png")
        output_image_path = Path(f"{self.data_folder}/{update.effective_user.id}_generated_sticker.png")
        if not input_image_path.exists():
            await update.effective_message.reply_text("Please, try again")
            self.logger.error("Error occured during sticker generation : No data is provided.")
            return await self.restart(update, context)
        input_image = Image.open(input_image_path).convert('RGB')
        result_image = self.sticker_processor.generate_sticker(input_image)
        if not result_image:
            await update.effective_message.reply_text("An error occured during sticker generation")
            self.logger.error("Error occured during sticker generation : No sticker is generated.")
            return await self.restart(update, context)
        result_image.save(output_image_path)
        with open(output_image_path, "rb") as photo_file:
            await update.effective_message.reply_document(photo_file, filename=f"edited_sticker_{update.effective_user.username}")
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

        await update.effective_message.reply_text("Audio to text placeholder")
        return await self.audio_to_text_continue(update, context)

    async def edit_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
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
        if update.effective_user:
            path = f"{self.data_folder}/{update.effective_user.id}_edit.png"
        else:
            path = f"{self.data_folder}/unknown_user_id_edit.png"
        await photo.download_to_drive(path)
        # temporarily echo the file
        with open(path, "rb") as photo_file:
            await update.effective_message.reply_photo(photo_file)
        return await self.edit_photo_continue(update, context)

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
        Handler to process more photos to sticker or return to menu
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
        Handler to process more photos to sticker or return to menu
        """
        self.logger.info("EVENT: edit_photo contunue")
        menu_keyboard = InlineKeyboardMarkup(PROMPT_IF_CONTINUE_TRANSCRIBE)

        if update.message:
            self.logger.info("Rendered keyboard in continue_edit")
            await update.message.reply_text(
                "Do you want to prepare more audios?",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in edit_photo_continue")

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

