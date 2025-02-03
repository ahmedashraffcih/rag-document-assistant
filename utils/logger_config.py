import os
import logging
import logging.config
import time
from datetime import datetime
from typing import Callable, Any, TypeVar, cast
import functools

F = TypeVar("F", bound=Callable[..., Any])


class LoggerConfig:
    """
    Handles logging configuration and provides utility decorators for logging execution and timing.
    """

    def __init__(
        self, config_filename: str = "logging.dev.ini", log_dir: str = "logs/"
    ) -> None:
        """
        Initializes LoggerConfig with paths and sets up logging.

        Args:
            config_filename (str): The logging configuration file.
            log_dir (str): The directory where log files will be stored.
        """
        self.base_dir = os.path.dirname(
            os.path.dirname(__file__)
        ) 
        self.config_dir = os.path.join(self.base_dir, "config")  # Config folder path
        self.log_dir = os.path.join(self.base_dir, log_dir)  # Log directory
        self.config_path = os.path.join(self.config_dir, config_filename)

        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging
        self.log_filename = self._setup_logging()

    def _setup_logging(self) -> str:
        """Loads logging configuration and creates a log file with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(self.log_dir, f"{timestamp}.log")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Logging configuration file not found: {self.config_path}"
            )

        logging.config.fileConfig(
            self.config_path,
            disable_existing_loggers=False,
            defaults={"logfilename": log_filename},
        )

        self._set_external_logging_levels()

        return log_filename

    @staticmethod
    def _set_external_logging_levels() -> None:
        """Reduces log noise from third-party libraries."""
        suppress_logs = [
            "httpx",
            "httpcore",
            "urllib3",
            "transformers",
            "pdfminer",
            "_trace",
            "psparser",
            "pdfinterp",
            "local_persistent_hnsw",
        ]
        for logger_name in suppress_logs:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger("local_persistent_hnsw").setLevel(logging.ERROR)


# Initialize logging
logger_config = LoggerConfig()
logger = logging.getLogger(__name__)


def log_execution(func: F) -> F:
    """
    A decorator to log the execution of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with added logging.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"🔄 Started {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ Finished {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise

    return cast(F, wrapper)  # Explicitly cast wrapper to the same type as func


def timing_decorator(func: F) -> F:
    """
    A decorator to log the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with added timing logic.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"⏳ {func.__name__} took {elapsed_time:.4f} seconds")
        return result

    return cast(F, wrapper)  # Explicitly cast wrapper to the same type as func
