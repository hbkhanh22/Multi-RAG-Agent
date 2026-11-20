"""
Helper utilities.
"""

import time

from src.utils.logger import app_logger


class Timer:
    """Context manager that logs how long a block took."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        app_logger.info(f"{self.name} completed in {self.end_time - self.start_time:.2f} seconds")

    def duration(self) -> float:
        """Duration in seconds (0 while running)."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
