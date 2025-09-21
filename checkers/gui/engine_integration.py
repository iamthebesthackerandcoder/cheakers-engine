"""
AI engine integration for the Checkers GUI.
"""
from __future__ import annotations

from typing import Optional, Tuple, Callable, Any
import threading
import time

from checkers.engine import get_engine
from checkers.eval import get_neural_evaluator


class EngineIntegration:
    """Handles AI engine integration and thinking."""

    def __init__(self):
        self.engine = get_engine()
        self.engine_depth = 6
        self.is_thinking = False
        self.use_neural_eval = False

    def set_engine_depth(self, depth: int) -> None:
        """Set the search depth for the engine."""
        self.engine_depth = max(1, min(10, depth))

    def set_neural_evaluation(self, enabled: bool) -> None:
        """Enable or disable neural network evaluation."""
        self.use_neural_eval = enabled
        if enabled:
            evaluator = get_neural_evaluator()
            self.engine.neural_evaluator = evaluator
        else:
            self.engine.neural_evaluator = None

    def search_async(self, board: Any, player: int, depth: int,
                    callback: Callable[[Optional[float], Optional[List[int]], float], None]) -> None:
        """Perform asynchronous engine search."""
        if self.is_thinking:
            return

        self.is_thinking = True
        board_copy = board.copy()
        search_player = player
        search_depth = max(1, min(10, depth))

        def worker():
            start_time = time.time()
            value, move = self.engine.search(board_copy, search_player, search_depth)
            elapsed_time = time.time() - start_time

            # Schedule callback on main thread
            from tkinter import messagebox
            import tkinter as tk
            root = tk._get_default_root()
            if root:
                root.after(0, lambda: callback(value, move, elapsed_time))

        threading.Thread(target=worker, daemon=True).start()

    def get_engine_move_async(self, board: Any, player: int,
                             on_complete: Callable[[Optional[float], Optional[List[int]], float], None]) -> None:
        """Get engine move asynchronously."""
        self.search_async(board, player, self.engine_depth, on_complete)

    def get_hint_async(self, board: Any, player: int,
                      on_complete: Callable[[Optional[float], Optional[List[int]], float], None]) -> None:
        """Get hint move asynchronously with reduced depth."""
        hint_depth = max(2, min(8, self.engine_depth))
        self.search_async(board, player, hint_depth, on_complete)
