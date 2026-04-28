# extern imports
from transformers import pipeline

# local imports
from .defines import TRANSLATION_MODEL


class Translator:
    def __init__(self) -> None:
        self.translator = pipeline(task="translation", model=TRANSLATION_MODEL)

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.split())

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        normalized: str = self._normalize(text)
        result = self.translator(normalized, max_length=512)
        return str(result[0]["translation_text"])
