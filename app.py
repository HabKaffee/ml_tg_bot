import logging
import os
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ConversationHandler, MessageHandler, filters

from src.image_processing.image_processor import ImageProcessor
from src.sticker_generator.sticker_generator import StickerGenerator
from src.telegram_bot.bot import TelegramBot
from src.telegram_bot.utils import BOT_STATES

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    Path("data/").mkdir(parents=True, exist_ok=True)
    Path("models/").mkdir(parents=True, exist_ok=True)

    app = Application.builder().token(os.environ["BOT_TOKEN"]).build()
    image_processor = ImageProcessor()
    sticker_generator = StickerGenerator()

    bot = TelegramBot("", image_processor, sticker_generator, logger)
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
