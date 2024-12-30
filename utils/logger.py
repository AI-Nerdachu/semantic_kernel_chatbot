import logging
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Sets up a logging instance with console and optional file logging.

    Args:
        log_level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file (bool): If True, logs are also written to a file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Convert log level string to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("semantic-kernel")
    logger.setLevel(level)

    # Prevent duplicate handlers if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for log messages
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        # Ensure the logs directory exists
        base_dir = os.path.abspath(os.path.dirname(__file__))
        logs_dir = os.path.join(base_dir, "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Log file path
        log_file = os.path.join(logs_dir, f"semantic-kernel-{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
