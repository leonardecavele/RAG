# extern
from transformers import pipeline
from langdetect import DetectorFactory, LangDetectException, detect

# local
from ..defines import TRANSLATION_MODEL

DetectorFactory.seed = 0


class Translator:
    """Translate user queries into English when needed."""

    def __init__(self) -> None:
        """Load the translation pipeline."""

        self.translator = pipeline(task="translation", model=TRANSLATION_MODEL)

    @staticmethod
    def _normalize(query: str) -> str:
        """Collapse whitespace in a query."""

        return " ".join(query.split())

    def _is_english(self, text: str) -> bool:
        """Return whether text is detected as English."""

        try:
            return bool(detect(text) == "en")
        except LangDetectException:
            return True

    def translate_to_english(self, text: str) -> str:
        """Translate text to English unless it is already English."""

        if not text.strip():
            return ""

        normalized: str = self._normalize(text)

        if self._is_english(text):
            return normalized

        result = self.translator(normalized, max_length=512)
        return str(result[0]["translation_text"])
