import os
import sys
from pathlib import Path

from loguru import logger

_level = os.getenv("LOG_LEVEL", "INFO").upper()

# src/mystic_forge/logger.py → parents[2] = project root
_ROOT = Path(__file__).parents[2]

_FMT_NORMAL      = "<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | <level>{message}</level>\n"
_FMT_FILE_NORMAL = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n"


def _relpath(r) -> str:
    try:
        rel = Path(r["file"].path).relative_to(_ROOT)
    except ValueError:
        rel = Path(r["file"].path)
    path = str(rel).replace("<", "\\<").replace(">", "\\>")
    return f"{path}:{r['line']}"


def _fmt_console(r):
    if r["level"].no < logger.level("WARNING").no:
        return _FMT_NORMAL
    loc = _relpath(r)
    return f"<dim>{{time:YYYY-MM-DD HH:mm:ss}}</dim> | <level>{{message}}</level> <dim>({loc})</dim>\n"


def _fmt_file(r):
    if r["level"].no < logger.level("WARNING").no:
        return _FMT_FILE_NORMAL
    loc = _relpath(r)
    return f"{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {{message}} ({loc})\n"


logger.remove()
logger.add(sys.stderr, level=_level, colorize=True, format=_fmt_console)
logger.add(
    "logs/mystic_forge.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    colorize=False,
    format=_fmt_file,
)
