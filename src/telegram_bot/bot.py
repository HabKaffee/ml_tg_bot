from logging import Logger
from typing import Any

from telegram import InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from src.image_processing.image_processor import ImageProcessor
from src.sticker_generator.sticker_generator import StickerGenerator

from .utils import BOT_STATES, KEYBOARD


class TelegramBot:
    def __init__(
        self,
        audio_processor: Any,
        image_processor: ImageProcessor,
        sticker_processor: StickerGenerator,
        logger: Logger,
        data_folder="data/",
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
        menu_keyboard = InlineKeyboardMarkup(KEYBOARD)

        if update.message:
            await update.message.reply_text(
                "Hi! Welcome to the bot. Choose the command to continue",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Exception in start")

        return BOT_STATES.ACTION_SELECTION

    async def photo_to_sticker_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a photo."""
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send a photo to prepare sticker.")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in photo_to_sticker_prompt")
        return BOT_STATES.PHOTO_STICKER

    async def edit_photo_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a photo."""
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            await query.message.reply_text("Please send a photo to edit.")  # type: ignore[union-attr]
        else:
            self.logger.error("Exception in edit_photo_prompt")
        return BOT_STATES.PHOTO_EDIT

    async def audio_to_text_prompt(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """Handles the button click and asks for a photo."""
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
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)
        if not update.effective_message.photo:
            await update.effective_message.reply_text("Please provide photo first.")
            return await self.restart(update, context)
        photo = await update.effective_message.photo[-1].get_file()
        if update.effective_user:
            path = f"{self.data_folder}/{update.effective_user.id}_sticker.png"
        else:
            path = f"{self.data_folder}/unknown_user_id_sticker.png"
        await photo.download_to_drive(path)
        # temporarily echo the file
        with open(path, "rb") as photo_file:
            await update.effective_message.reply_photo(photo_file)

        return await self.restart(update, context)

    async def audio_to_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to gathed needed data and call audio to text conversion model
        """
        if not update.effective_message:
            self.logger.warning("Update.effective_message is empty")
            return await self.restart(update, context)

        await update.effective_message.reply_text("Audio to text placeholder")
        return await self.restart(update, context)

    async def edit_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        """
        Handler to gathed needed data and call picture edit model
        """
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
        return await self.restart(update, context)

    async def cancel(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancels the process and returns to the main menu."""
        if update.effective_message:
            await update.effective_message.reply_text("Okay, process canceled. Type /start to restart.")
        else:
            self.logger.error("Exception in cancel")
        return ConversationHandler.END

    async def restart(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> BOT_STATES:
        menu_keyboard = InlineKeyboardMarkup(KEYBOARD)
        if update.message:
            await update.message.reply_text(
                "Choose the command to continue",
                reply_markup=menu_keyboard,
            )
        else:
            self.logger.error("Can't render keyboard")
        return BOT_STATES.ACTION_SELECTION
