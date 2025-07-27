"""
Logging utility for the voice assistant.
"""

import datetime
import os
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init()

LOG_LEVEL = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4
}

CURRENT_LOG_LEVEL = LOG_LEVEL["INFO"]

# Color mappings
LOG_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT,
    "SPEECH": Fore.MAGENTA + Style.BRIGHT
}

def get_timestamp():
    """Return current timestamp in a readable format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message, level="INFO"):
    """
    Log a message with timestamp and level.
    
    Args:
        message (str): The message to log
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, SPEECH)
    """
    if level == "SPEECH" or LOG_LEVEL.get(level, 0) >= CURRENT_LOG_LEVEL:
        timestamp = get_timestamp()
        color = LOG_COLORS.get(level, Fore.WHITE)
        
        if level == "SPEECH":
            # Special format for speech output
            log_entry = f"{color}🔊 SPEAKING: \"{message}\"{Style.RESET_ALL}"
            # Use stderr for immediate output without buffering
            print(log_entry, flush=True)
        else:
            # Standard log format
            log_entry = f"{color}[{timestamp}] [{level}] {message}{Style.RESET_ALL}"
            print(log_entry, flush=True)
        
        # Optional: Write to log file
        # with open("assistant_log.txt", "a") as log_file:
        #     log_file.write(f"[{timestamp}] [{level}] {message}\n")

def debug(message):
    log(message, "DEBUG")

def info(message):
    log(message, "INFO")

def warning(message):
    log(message, "WARNING")

def error(message):
    log(message, "ERROR")

def critical(message):
    log(message, "CRITICAL")

def speech(message):
    """Special log for speech output with high visibility."""
    log(message, "SPEECH")
