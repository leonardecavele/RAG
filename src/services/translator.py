# extern
from transformers import pipeline
from langdetect import DetectorFactory, LangDetectException, detect

# local
from ..defines import TRANSLATION_MODEL

DetectorFactory.seed = 0


class Translator:
    def __init__(self) -> None:
        self.translator = pipeline(task="translation", model=TRANSLATION_MODEL)

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.split())

    def _is_english(self, text: str) -> bool:
        try:
            return bool(detect(text) == "en")
        except LangDetectException:
            return True

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        normalized: str = self._normalize(text)

        if self._is_english(text):
            return normalized

        result = self.translator(normalized, max_length=512)
        return str(result[0]["translation_text"])
