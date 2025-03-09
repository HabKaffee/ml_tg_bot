import os
import logging

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update

from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

ACTION_SELECTION, PHOTO_STICKER, PHOTO_EDIT, AUDIO = range(4)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Starts the conversation
    """

    reply_keywords = [["Convert audio to text", "Edit the picture", "Create sticker pack"]]

    await update.message.reply_text(
        "Hi! Welcome to the bot. Choose the command to continue",
        reply_markup=ReplyKeyboardMarkup(
            reply_keywords, input_field_placeholder="Action"
        )
    )

    return ACTION_SELECTION

async def audio_to_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Audio to text placeholder"
    )
    return AUDIO

async def photo_to_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Photo to sticker placeholder"
    )
    return PHOTO_STICKER

async def edit_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Edit photo placeholder"
    )
    return PHOTO_EDIT

async def exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("All clear!")

def main() -> None:
    app = Application.builder().token(os.environ['BOT_TOKEN']).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("audio", audio_to_text))
    app.add_handler(CommandHandler("sticker", photo_to_sticker))
    app.add_handler(CommandHandler("edit", edit_photo))
    app.add_handler(CommandHandler("exit", exit))


    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
