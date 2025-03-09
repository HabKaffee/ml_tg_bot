import os
import logging

from telegram import CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update

from telegram.ext import (
    Application,
    CallbackQueryHandler,
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

logger = logging.getLogger(__name__)

ACTION_SELECTION, PHOTO_STICKER, PHOTO_EDIT, AUDIO = range(4)

# async def menu_keyboard() -> InlineKeyboardMarkup:
#     reply_keywords = [
#         InlineKeyboardButton("Convert audio to text", callback_data="audio"),
#         InlineKeyboardButton("Edit the picture", callback_data="edit"),
#         InlineKeyboardButton("Create sticker pack", callback_data="sticker")
#     ]
#     return InlineKeyboardMarkup(reply_keywords)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Starts the conversation
    """
    keyboard = [
        [InlineKeyboardButton("Convert audio to text", callback_data="audio")],
        [InlineKeyboardButton("Edit the picture", callback_data="edit")],
        [InlineKeyboardButton("Create sticker pack", callback_data="sticker")],
    ]
    menu_keyboard = InlineKeyboardMarkup(keyboard)


    await update.message.reply_text(
        "Hi! Welcome to the bot. Choose the command to continue",
        reply_markup=menu_keyboard,
    )

    return ACTION_SELECTION


async def audio_to_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call audio to text conversion model
    """
    await update.effective_message.reply_text(
        "Audio to text placeholder"
    )
    return AUDIO


async def photo_to_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call sticker pack creation model
    """
    await update.effective_message.reply_text(
        "Photo to sticker placeholder"
    )
    return PHOTO_STICKER


async def edit_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handler to gathed needed data and call picture edit model
    """
    await update.effective_message.reply_text(
        "Edit photo placeholder"
    )
    return PHOTO_EDIT


async def exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler to exit
    """
    await update.effective_message.reply_text("All clear!")


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler for button reply
    """
    logger.info("Button handler")
    query = update.callback_query
    await query.answer()
    
    await update.effective_message.reply_text(f"Selected {query.data}")

    if query.data == "audio":
        await audio_to_text(update, context)
    elif query.data == "sticker":
        await photo_to_sticker(update, context)
    elif query.data == "edit":
        await edit_photo(update, context)
    elif query.data == "exit":
        await exit(update, context)
    else:
        print(f"unknown command: {query.data}")


def main() -> None:
    app = Application.builder().token(os.environ['BOT_TOKEN']).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("audio", audio_to_text))
    app.add_handler(CommandHandler("sticker", photo_to_sticker))
    app.add_handler(CommandHandler("edit", edit_photo))
    app.add_handler(CommandHandler("exit", exit))
    app.add_handler(CallbackQueryHandler(button))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
