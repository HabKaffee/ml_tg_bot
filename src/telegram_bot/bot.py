import os
import logging

from telegram import (
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

ACTION_SELECTION, PHOTO_STICKER, PHOTO_EDIT, AUDIO = range(4)

KEYBOARD = [
    [InlineKeyboardButton("Convert audio to text", callback_data="audio")],
    [InlineKeyboardButton("Edit the picture", callback_data="edit")],
    [InlineKeyboardButton("Create sticker pack", callback_data="sticker")],
    [InlineKeyboardButton("Cancel", callback_data="cancel")],
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Starts the conversation
    """
    menu_keyboard = InlineKeyboardMarkup(KEYBOARD)

    await update.message.reply_text(
        "Hi! Welcome to the bot. Choose the command to continue",
        reply_markup=menu_keyboard,
    )

    return ACTION_SELECTION


async def photo_to_sticker_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the button click and asks for a photo."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Please send a photo to prepare sticker.")

    return PHOTO_STICKER


async def edit_photo_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the button click and asks for a photo."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Please send a photo to edit.")

    return PHOTO_EDIT


async def audio_to_text_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the button click and asks for a photo."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Please send audio to transcribe.")

    return AUDIO


async def photo_to_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call sticker pack creation model
    """
    if not update.effective_message.photo:
        await update.effective_message.reply_text("Please provide photo first.")
        return await restart(update, context)

    photo = await update.effective_message.photo[-1].get_file()
    path = f"data/{update.effective_user.id}_sticker.png"
    await photo.download_to_drive(path)
    # temporarily echo the file
    with open(path, "rb") as photo_file:
        await update.effective_message.reply_photo(photo_file)

    return await restart(update, context)


async def audio_to_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call audio to text conversion model
    """
    await update.effective_message.reply_text("Audio to text placeholder")
    return await restart(update, context)


async def edit_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call picture edit model
    """
    if not update.effective_message.photo:
        await update.effective_message.reply_text("Please provide photo first.")
        return await restart(update, context)

    photo = await update.effective_message.photo[-1].get_file()
    path = f"data/{update.effective_user.id}_edit.png"
    await photo.download_to_drive(path)
    # temporarily echo the file
    with open(path, "rb") as photo_file:
        await update.effective_message.reply_photo(photo_file)
    return await restart(update, context)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the process and returns to the main menu."""
    await update.effective_message.reply_text("Okay, process canceled. Type /start to restart.")
    return ConversationHandler.END


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    menu_keyboard = InlineKeyboardMarkup(KEYBOARD)
    await update.message.reply_text(
        "Choose the command to continue",
        reply_markup=menu_keyboard,
    )
    return ACTION_SELECTION


def main() -> None:
    if not os.path.isdir("data/"):
        logger.info("Creating data directory for files.")
        os.mkdir("data/")

    app = Application.builder().token(os.environ["BOT_TOKEN"]).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ACTION_SELECTION: [
                CallbackQueryHandler(audio_to_text_prompt, pattern="^audio"),
                CallbackQueryHandler(edit_photo_prompt, pattern="^edit$"),
                CallbackQueryHandler(photo_to_sticker_prompt, pattern="^sticker"),
                CallbackQueryHandler(cancel, pattern="^cancel"),
            ],
            PHOTO_EDIT: [
                MessageHandler(filters.PHOTO, edit_photo),
            ],
            PHOTO_STICKER: [
                MessageHandler(filters.PHOTO, photo_to_sticker),
            ],
            AUDIO: [
                MessageHandler(filters.VOICE, audio_to_text),
            ],
        },
        fallbacks=[CommandHandler("restart", restart)],
        allow_reentry=True,
        conversation_timeout=3600,
    )

    app.add_handler(conv_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
