"""
GUI component factory for creating and configuring GUI components.
"""
from __future__ import annotations

from typing import Tuple, Any
import tkinter as tk
from tkinter import ttk

from checkers.gui.constants import BOARD_SIZE, SQUARE_SIZE, FONT_TITLE, FONT_NORMAL, FONT_LABEL
from checkers.gui.game_state import GameState
from checkers.gui.board_renderer import BoardRenderer
from checkers.gui.move_manager import MoveManager
from checkers.gui.engine_integration import EngineIntegration
from checkers.gui.training_integration import TrainingIntegration


class GUIComponentFactory:
    """Factory for creating and configuring GUI components."""

    @staticmethod
    def create_main_canvas(parent: ttk.Frame) -> tk.Canvas:
        """Create the main board canvas."""
        canvas = tk.Canvas(
            parent,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            highlightthickness=0,
            bg="#EEEED2"  # BOARD_BG_LIGHT
        )
        return canvas

    @staticmethod
    def create_controls_frame(parent: ttk.Frame, padding: Tuple[int, int] = (10, 0)) -> ttk.Frame:
        """Create the controls frame."""
        return ttk.Frame(parent, padding=padding)

    @staticmethod
    def create_button(parent: ttk.Frame, text: str, command: Any) -> ttk.Button:
        """Create a styled button."""
        return ttk.Button(parent, text=text, command=command)

    @staticmethod
    def create_label(parent: Any, text: str, font: Tuple[str, int, str] = FONT_LABEL,
                    **kwargs) -> ttk.Label:
        """Create a styled label."""
        return ttk.Label(parent, text=text, font=font, **kwargs)

    @staticmethod
    def create_scale(parent: ttk.Frame, from_: int, to: int, variable: tk.Variable,
                    orient: str = "horizontal", **kwargs) -> ttk.Scale:
        """Create a styled scale widget."""
        return ttk.Scale(parent, from_=from_, to=to, orient=orient, variable=variable, **kwargs)

    @staticmethod
    def create_combobox(parent: ttk.Frame, values: list, textvariable: tk.StringVar,
                       state: str = "readonly", width: int = 10) -> ttk.Combobox:
        """Create a styled combobox."""
        combo = ttk.Combobox(parent, values=values, state=state, textvariable=textvariable, width=width)
        return combo

    @staticmethod
    def create_checkbutton(parent: ttk.Frame, text: str, variable: tk.Variable,
                          command: Any = None) -> ttk.Checkbutton:
        """Create a styled checkbutton."""
        return ttk.Checkbutton(parent, text=text, variable=variable, command=command)

    @staticmethod
    def create_listbox(parent: ttk.Frame, height: int = 10, width: int = 28,
                      **kwargs) -> tk.Listbox:
        """Create a styled listbox."""
        return tk.Listbox(parent, height=height, width=width, activestyle="dotbox", **kwargs)

    @staticmethod
    def create_status_frame(parent: ttk.Frame, padding: Tuple[int, int] = (0, 8)) -> ttk.Frame:
        """Create the status frame."""
        return ttk.Frame(parent, padding=padding)


class GUIFactory:
    """Main factory for creating all GUI components with dependencies."""

    @staticmethod
    def create_game_state() -> GameState:
        """Create a new game state."""
        return GameState()

    @staticmethod
    def create_board_renderer(canvas: tk.Canvas) -> BoardRenderer:
        """Create a board renderer with the given canvas."""
        return BoardRenderer(canvas)

    @staticmethod
    def create_move_manager(board: Any, current_player: int) -> MoveManager:
        """Create a move manager with initial game state."""
        return MoveManager(board, current_player)

    @staticmethod
    def create_engine_integration() -> EngineIntegration:
        """Create engine integration."""
        return EngineIntegration()

    @staticmethod
    def create_training_integration(parent: tk.Tk) -> TrainingIntegration:
        """Create training integration with parent window."""
        return TrainingIntegration(parent)
