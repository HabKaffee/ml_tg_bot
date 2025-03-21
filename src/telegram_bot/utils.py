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

PROMPT_IF_CONTINUE_STICKER = [
    [InlineKeyboardButton("Process more photos", callback_data="continue_sticker")],
    [InlineKeyboardButton("Return to menu", callback_data="return")]
]

PROMPT_IF_CONTINUE_EDIT = [
    [InlineKeyboardButton("Process more photos", callback_data="continue_edit")],
    [InlineKeyboardButton("Return to menu", callback_data="return")]
]

PROMPT_IF_CONTINUE_TRANSCRIBE = [
    [InlineKeyboardButton("Process more audio", callback_data="continue_audio")],
    [InlineKeyboardButton("Return to menu", callback_data="return")]
]
