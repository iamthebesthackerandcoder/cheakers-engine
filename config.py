"""
Central configuration for paths and tunables.
"""
from __future__ import annotations

import os
import logging

# Directories
DATA_DIR = os.environ.get("CHECKERS_DATA_DIR", os.path.join("data"))
MODELS_DIR = os.environ.get("CHECKERS_MODELS_DIR", os.path.join("models"))

# Default artifact paths
DEFAULT_MODEL_PATH = os.path.join(DATA_DIR, "neural_model.pkl")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "training_data.pkl")  # will be saved as .npz
DEFAULT_CHECKPOINT_PATH = os.path.join(DATA_DIR, "training_checkpoint.pkl")

# Training exploration schedule
EPSILON_START = float(os.environ.get("CHECKERS_EPS_START", 0.10))
EPSILON_END = float(os.environ.get("CHECKERS_EPS_END", 0.01))
EPSILON_DECAY_GAMES = int(os.environ.get("CHECKERS_EPS_DECAY", 100))

# Search defaults
DEFAULT_SEARCH_DEPTHS = (4, 5, 6)

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.environ.get("CHECKERS_LOG_LEVEL", "INFO").upper()


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR):
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def setup_logging() -> None:
    """Configure root logging once, controlled by env var CHECKERS_LOG_LEVEL."""
    if getattr(setup_logging, "_configured", False):
        return
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    setup_logging._configured = True  # type: ignore[attr-defined]
