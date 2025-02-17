import numpy as np
from numpy.typing import NDArray


class TelegramMockAPI:
    """
    Mocked version of the Telegram API
    """

    @staticmethod
    def receive_message() -> str:
        text = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz "), size=20))
        return text

    @staticmethod
    def receive_photo() -> NDArray[np.float32]:
        photo = np.random.rand(256, 256, 3).astype(np.float32)
        return photo
