import os
import logging

from bot import TelegramBot
from utils import BOT_STATES

from telegram import Update

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    if not os.path.isdir("data/"):
        logger.info("Creating data directory for files.")
        os.mkdir("data/")

    app = Application.builder().token(os.environ["BOT_TOKEN"]).build()
    bot = TelegramBot("", "", "", logger)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", bot.start)],
        states={
            BOT_STATES.ACTION_SELECTION: [
                CallbackQueryHandler(bot.audio_to_text_prompt, pattern="^audio"),
                CallbackQueryHandler(bot.edit_photo_prompt, pattern="^edit$"),
                CallbackQueryHandler(bot.photo_to_sticker_prompt, pattern="^sticker"),
                CallbackQueryHandler(bot.cancel, pattern="^cancel"),
            ],
            BOT_STATES.PHOTO_EDIT: [
                MessageHandler(filters.PHOTO, bot.edit_photo),
            ],
            BOT_STATES.PHOTO_STICKER: [
                MessageHandler(filters.PHOTO, bot.photo_to_sticker),
            ],
            BOT_STATES.AUDIO: [
                MessageHandler(filters.VOICE, bot.audio_to_text),
            ],
        },
        fallbacks=[CommandHandler("restart", bot.restart)],
        allow_reentry=True,
        conversation_timeout=3600,
    )

    app.add_handler(conv_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
