class LoggerWriter:
    """A writer class that redirects writes to a logger instance.
    
    This class is used to redirect stdout and stderr to the logging system,
    allowing all print statements and error messages to be captured in the log file.
    
    Attributes:
        level: The logging level function to use (e.g., logging.info, logging.error).
    """
    
    def __init__(self, level):
        """Initialize with the given logging level function."""
        self.level = level
        self.buffer = ""

    def write(self, message):
        """Write the message to the logger."""
        if message and not message.isspace():
            # If the message doesn't end with a newline, buffer it
            if message.endswith('\n'):
                self.level(self.buffer + message.rstrip('\n'))
                self.buffer = ""
            else:
                self.buffer += message

    def flush(self):
        """Flush the buffer to the logger."""
        if self.buffer:
            self.level(self.buffer)
            self.buffer = ""

