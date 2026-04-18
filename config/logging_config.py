"""
config/logging_config.py
─────────────────────────
Centralised logger for ARIA.
Import anywhere:  from config.logging_config import logger
"""

import logging
import sys
from pathlib import Path

# ── Log file ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("data")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "aria.log"

# ── Formatter ─────────────────────────────────────────────────────────────────
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Handlers ──────────────────────────────────────────────────────────────────
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# ── Root logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("ARIA")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
