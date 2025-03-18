from enum import IntEnum

from telegram import InlineKeyboardButton


class BOT_STATES(IntEnum):
    ACTION_SELECTION = 0
    PHOTO_STICKER = 1
    PHOTO_EDIT = 2
    AUDIO = 3


KEYBOARD = [
    [InlineKeyboardButton("Convert audio to text", callback_data="audio")],
    [InlineKeyboardButton("Edit the picture", callback_data="edit")],
    [InlineKeyboardButton("Create sticker pack", callback_data="sticker")],
    [InlineKeyboardButton("Cancel", callback_data="cancel")],
]
