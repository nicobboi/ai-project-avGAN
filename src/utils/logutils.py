import enum
from datetime import datetime

class ColorEnum(enum.StrEnum):
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"

def info(text):
    print(f"{_build_message('INFO', text)}")

def success(text):
    print(f"{ColorEnum.GREEN}{_build_message('SUCCESS', text)}{ColorEnum.RESET}")

def error(text):
    print(f"{ColorEnum.RED}{_build_message('ERROR', text)}{ColorEnum.RESET}")

def warning(text):
    print(f"{ColorEnum.YELLOW}{_build_message('WARNING', text)}{ColorEnum.RESET}")

def _build_message(level, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] [{level}] {text}"