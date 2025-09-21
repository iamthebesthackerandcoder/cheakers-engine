"""
GUI components for the Checkers application.
"""
from __future__ import annotations

from .constants import *
from .game_state import GameState
from .board_renderer import BoardRenderer
from .move_manager import MoveManager, group_moves_by_start, group_moves_by_dest, manhattan_distance
from .engine_integration import EngineIntegration
from .training_integration import TrainingIntegration
from .factory import GUIComponentFactory, GUIFactory
from .checkers_ui import CheckersUI

__all__ = [
    # Constants
    "BOARD_BG_LIGHT", "BOARD_BG_DARK", "BOARD_HL_SQ", "BOARD_HL_DEST",
    "BOARD_LASTMOVE", "PIECE_BLACK_FILL", "PIECE_BLACK_OUTLINE",
    "PIECE_RED_FILL", "PIECE_RED_OUTLINE", "KING_TEXT_COLOR",
    "SQUARE_SIZE", "BOARD_SIZE", "FONT_TITLE", "FONT_NORMAL",
    "FONT_LABEL", "FONT_BUTTON", "TEXT_COLOR_NORMAL",
    "TEXT_COLOR_DISABLED", "HINT_COLOR",

    # Core components
    "GameState",
    "BoardRenderer",
    "MoveManager",
    "EngineIntegration",
    "TrainingIntegration",

    # Factories
    "GUIComponentFactory",
    "GUIFactory",

    # Main UI
    "CheckersUI",

    # Utility functions
    "group_moves_by_start",
    "group_moves_by_dest",
    "manhattan_distance",
]
