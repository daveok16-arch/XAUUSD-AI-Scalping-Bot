"""
Logging utility for the XAUUSD Scalping Bot.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "xauusd_bot", level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
