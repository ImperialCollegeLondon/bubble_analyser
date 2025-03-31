"""Logger redirection utilities for capturing output to logging system.

This module provides functionality to redirect standard output and error streams
to the Python logging system, ensuring that all console output is properly captured
in log files with appropriate logging levels.

It's particularly useful for capturing print statements and error messages in
applications where logging is preferred over console output.
"""

from collections.abc import Callable


class LoggerWriter:
    """A writer class that redirects writes to a logger instance.

    This class is used to redirect stdout and stderr to the logging system,
    allowing all print statements and error messages to be captured in the log file.

    Attributes:
        level: The logging level function to use (e.g., logging.info, logging.error).
    """

    def __init__(self, level: Callable[[str], None]):
        """Initialize with the given logging level function.

        Args:
            level: The logging level function to use (e.g., logging.info, logging.error).
        """
        self.level = level
        self.buffer = ""

    def write(self, message: str) -> int:  # type: ignore
        """Write the message to the logger.

        Args:
            message: The message string to be written to the logger.

        Returns:
            int: The number of characters written.
        """
        if not message:
            return 0

        if message and not message.isspace():
            # If the message doesn't end with a newline, buffer it
            if message.endswith("\n"):
                self.level(self.buffer + message.rstrip("\n"))
                self.buffer = ""
            else:
                self.buffer += message

            return len(message)

    def flush(self) -> None:
        """Flush the buffer to the logger.

        This method is called to ensure any buffered content is written to the logger.
        """
        if self.buffer:
            self.level(self.buffer)
            self.buffer = ""
