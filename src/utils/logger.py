# standard imports
import logging


class LoggerManager:
    LOG_LEVELS: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    def __init__(self, level: str = "error") -> None:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(levelname)s: %(message)s",
        )
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.set(level)

    def set(self, level: str) -> None:
        normalized: str = self.normalize_level(level)
        self.logger.setLevel(self.LOG_LEVELS[normalized])

    def library_level(self, level: str) -> None:
        normalized: str = self.normalize_level(level)
        log_level: int = self.LOG_LEVELS[normalized]

        logging.getLogger().setLevel(log_level)

        for logger_name in logging.root.manager.loggerDict:
            if logger_name == self.logger.name:
                continue

            logging.getLogger(logger_name).setLevel(log_level)

        self.logger.setLevel(self.logger.level)

    def normalize_level(self, level: str) -> str:
        normalized: str = level.lower()
        if normalized not in self.LOG_LEVELS:
            raise ValueError(
                "invalid log level, expected one of: "
                f"{', '.join(self.LOG_LEVELS.keys())}"
            )
        return normalized
