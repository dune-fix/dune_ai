import os
import sys
import logging
from pathlib import Path

from config.settings import BASE_DIR

# Ensure logs directory exists
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
GENERAL_LOG_FILE = LOGS_DIR / "dune_ai.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"

# Log formats
CONSOLE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


# Configure logging
def configure_logging():
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # File handlers
    general_file_handler = logging.FileHandler(GENERAL_LOG_FILE)
    general_file_handler.setLevel(logging.INFO)
    general_file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
    root_logger.addHandler(general_file_handler)

    error_file_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
    root_logger.addHandler(error_file_handler)

    # Create loggers for each module
    loggers = {
        "twitter_sentinel": setup_module_logger("twitter_sentinel"),
        "spice_trend_engine": setup_module_logger("spice_trend_engine"),
        "sandworm_scanner": setup_module_logger("sandworm_scanner"),
        "solana_client": setup_module_logger("solana_client"),
        "analytics": setup_module_logger("analytics"),
    }

    return loggers


def setup_module_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Module-specific log file
    module_log_file = LOGS_DIR / f"{module_name}.log"
    file_handler = logging.FileHandler(module_log_file)
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


# Get logger for a specific module
def get_logger(module_name):
    return logging.getLogger(module_name)