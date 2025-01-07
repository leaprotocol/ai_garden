import logging
import uuid

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A configured logging.Logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

def generate_cache_id() -> str:
    """
    Generates a unique cache ID.

    Returns:
        A unique cache ID string.
    """
    return str(uuid.uuid4()) 