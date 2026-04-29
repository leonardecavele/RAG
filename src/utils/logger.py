# standard
import logging

# extern
from rich.logging import RichHandler


class LoggerManager:
    """Configure application and library log levels."""

    LOG_LEVELS: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    def __init__(self, level: str = "error") -> None:
        """Create a logger manager with the requested app level."""

        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_time=False,
                    show_path=False,
                    markup=False,
                )
            ],
            force=True,
        )
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.set(level)

    def set(self, level: str) -> None:
        """Set the application logger level."""

        normalized: str = self.normalize_level(level)
        self.logger.setLevel(self.LOG_LEVELS[normalized])

    def library_level(self, level: str) -> None:
        """Set log levels for non-application loggers."""

        normalized: str = self.normalize_level(level)
        log_level: int = self.LOG_LEVELS[normalized]
        app_log_level: int = self.logger.level

        logging.getLogger().setLevel(log_level)

        for logger_name in logging.root.manager.loggerDict:
            if logger_name == self.logger.name:
                continue

            logging.getLogger(logger_name).setLevel(log_level)

        self.logger.setLevel(app_log_level)

    def normalize_level(self, level: str) -> str:
        """Normalize and validate a log level name."""

        normalized: str = level.lower()
        if normalized not in self.LOG_LEVELS:
            raise ValueError(
                "invalid log level, expected one of: "
                f"{', '.join(self.LOG_LEVELS.keys())}"
            )
        return normalized
