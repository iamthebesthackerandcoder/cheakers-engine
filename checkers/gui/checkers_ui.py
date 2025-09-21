"""
Refactored main Checkers GUI class using modular components.
"""
from __future__ import annotations

from typing import Optional, List
import tkinter as tk
from tkinter import messagebox, ttk

from checkers.engine import CAPTURES_MANDATORY
from checkers.gui.constants import (
    BOARD_SIZE, SQUARE_SIZE, FONT_TITLE, FONT_NORMAL, FONT_LABEL, FONT_BUTTON
)
from checkers.gui.factory import GUIComponentFactory, GUIFactory
from checkers.gui.game_state import GameState
from checkers.gui.board_renderer import BoardRenderer
from checkers.gui.move_manager import MoveManager
from checkers.gui.engine_integration import EngineIntegration
from checkers.gui.training_integration import TrainingIntegration


class CheckersUI(tk.Tk):
    """Main Checkers GUI class using modular components."""

    def __init__(self):
        super().__init__()
        self.title("Checkers - GUI")
        self.resizable(False, False)

        # Initialize components
        self.game_state = GUIFactory.create_game_state()
        self.engine_integration = GUIFactory.create_engine_integration()
        self.training_integration = GUIFactory.create_training_integration(self)

        # UI state
        self.selected_square: Optional[int] = None
        self.destination_map: dict = {}

        # Build UI
        self._build_ui()
        self._setup_event_handlers()
        self._new_game()

    def _build_ui(self):
        """Build the main UI layout."""
        container = ttk.Frame(self, padding=8)
        container.grid(row=0, column=0, sticky="nsew")

        # Left: Board
        self.canvas = GUIComponentFactory.create_main_canvas(container)
        self.canvas.grid(row=0, column=0, rowspan=3, sticky="n")

        # Right: Controls
        controls = GUIComponentFactory.create_controls_frame(container)
        controls.grid(row=0, column=1, sticky="nw")

        # Title
        title = GUIComponentFactory.create_label(controls, "Checkers", FONT_TITLE)
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Side selector
        self.human_side_var = tk.StringVar(value=self.game_state.human_side)
        GUIComponentFactory.create_label(controls, "You play as:").grid(row=1, column=0, sticky="w")
        side_combo = GUIComponentFactory.create_combobox(
            controls, ["Black", "Red"], self.human_side_var
        )
        side_combo.grid(row=1, column=1, sticky="w")
        side_combo.bind("<<ComboboxSelected>>", lambda e: self._new_game())

        # Difficulty
        self.engine_depth_var = tk.IntVar(value=self.engine_integration.engine_depth)
        GUIComponentFactory.create_label(controls, "Difficulty:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        depth_scale = GUIComponentFactory.create_scale(controls, 1, 10, self.engine_depth_var)
        depth_scale.grid(row=2, column=1, sticky="we", padx=(0, 4))
        self.depth_label = GUIComponentFactory.create_label(controls, f"Level {self.engine_depth_var.get()}")
        self.depth_label.grid(row=2, column=1, sticky="e")
        self.engine_depth_var.trace_add("write", self._on_depth_change)

        # Mandatory captures
        self.mandatory_var = tk.BooleanVar(value=CAPTURES_MANDATORY)
        mandatory_chk = GUIComponentFactory.create_checkbutton(
            controls, "Mandatory captures", self.mandatory_var, self._toggle_mandatory
        )
        mandatory_chk.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Neural eval toggle
        self.use_neural_var = tk.BooleanVar(value=self.engine_integration.use_neural_eval)
        nn_chk = GUIComponentFactory.create_checkbutton(
            controls, "Use Neural Eval", self.use_neural_var, self._toggle_use_neural
        )
        nn_chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Buttons
        btns = ttk.Frame(controls)
        btns.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 4))
        self.btn_new = GUIComponentFactory.create_button(btns, "New Game", self._new_game)
        self.btn_new.grid(row=0, column=0, padx=(0, 5))
        self.btn_undo = GUIComponentFactory.create_button(btns, "Undo", self._undo)
        self.btn_undo.grid(row=0, column=1, padx=5)
        self.btn_hint = GUIComponentFactory.create_button(btns, "Hint", self._hint)
        self.btn_hint.grid(row=0, column=2, padx=5)
        self.btn_flip = GUIComponentFactory.create_button(btns, "Flip Board", self._flip_board)
        self.btn_flip.grid(row=0, column=3, padx=5)
        self.btn_train = GUIComponentFactory.create_button(btns, "Train AI", self._start_training)
        self.btn_train.grid(row=0, column=4, padx=5)

        # Move list
        GUIComponentFactory.create_label(controls, "Your legal moves:").grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.moves_list = GUIComponentFactory.create_listbox(controls)
        self.moves_list.grid(row=7, column=0, columnspan=2, sticky="w")

        # Status panel
        status = GUIComponentFactory.create_status_frame(container)
        status.grid(row=2, column=1, sticky="nw")
        self.lbl_turn = GUIComponentFactory.create_label(status, "", FONT_NORMAL)
        self.lbl_turn.grid(row=0, column=0, sticky="w")
        self.lbl_counts = GUIComponentFactory.create_label(status, "")
        self.lbl_counts.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.lbl_eval = GUIComponentFactory.create_label(status, "", foreground="#888")
        self.lbl_eval.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Initialize board renderer
        self.board_renderer = GUIFactory.create_board_renderer(self.canvas)
        self.move_manager = GUIFactory.create_move_manager(
            self.game_state.board, self.game_state.current_player
        )

    def _setup_event_handlers(self):
        """Set up event handlers."""
        self.canvas.bind("<Button-1>", self.on_click)
        self.moves_list.bind("<Double-Button-1>", self._play_selected_move)

    def _new_game(self, _event=None):
        """Start a new game."""
        human_side = self.human_side_var.get()
        self.game_state.reset_game(human_side)

        # Update engine depth
        self.engine_integration.set_engine_depth(self.engine_depth_var.get())

        # Update move manager
        self.move_manager.set_game_state(self.game_state.board, self.game_state.current_player)

        # Update UI
        self._refresh_ui()

        # If engine plays first (human is Red), let engine move
        if not self.game_state.is_human_turn():
            if human_side == "Red":
                self.after(300, self._engine_move_async)

    def _refresh_ui(self):
        """Refresh all UI elements."""
        self._refresh_moves_list()
        self._update_status()
        self.board_renderer.redraw_board(self.game_state.board)

    def _refresh_moves_list(self):
        """Refresh the moves list display."""
        self.moves_list.delete(0, tk.END)
        if not self.move_manager.legal_moves_list:
            self.moves_list.insert(tk.END, "(no legal moves)")
            return

        for move_str in self.move_manager.get_move_display_strings():
            self.moves_list.insert(tk.END, move_str)

    def _update_status(self):
        """Update status labels."""
        self.lbl_turn.config(text=f"Move {self.game_state.move_number} — Turn: {self.game_state.get_current_player_name()}")
        blacks, reds, bk, rk = self.game_state.get_piece_counts()
        self.lbl_counts.config(text=f"Pieces: Black {blacks} (K:{bk}) | Red {reds} (K:{rk})")

    def _toggle_mandatory(self):
        """Toggle mandatory captures setting."""
        from checkers.engine import set_captures_mandatory
        set_captures_mandatory(self.mandatory_var.get())
        self._refresh_moves()

    def _toggle_use_neural(self):
        """Toggle neural network evaluation."""
        self.engine_integration.set_neural_evaluation(self.use_neural_var.get())
        self.lbl_eval.config(text="Neural evaluation enabled" if self.use_neural_var.get() else "Neural evaluation disabled")

    def _on_depth_change(self, *args):
        """Handle depth slider change."""
        depth = self.engine_depth_var.get()
        self.engine_integration.set_engine_depth(depth)
        self.depth_label.config(text=f"Level {depth}")

    def _refresh_moves(self):
        """Refresh legal moves."""
        self.move_manager.set_game_state(self.game_state.board, self.game_state.current_player)
        self._refresh_moves_list()
        self._update_status()

    def _flip_board(self):
        """Flip the board orientation."""
        self.board_renderer.set_board_orientation(not self.board_renderer.flipped)
        self._refresh_ui()

    def on_click(self, event):
