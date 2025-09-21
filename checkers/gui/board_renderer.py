"""
Board rendering functionality for the Checkers GUI.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Any
import tkinter as tk

from checkers.engine import rc, idx_map
from checkers.gui.constants import (
    BOARD_BG_LIGHT, BOARD_BG_DARK, BOARD_HL_SQ, BOARD_HL_DEST,
    BOARD_LASTMOVE, PIECE_BLACK_FILL, PIECE_BLACK_OUTLINE,
    PIECE_RED_FILL, PIECE_RED_OUTLINE, KING_TEXT_COLOR, SQUARE_SIZE
)


class BoardRenderer:
    """Handles all board rendering and visual updates."""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.flipped = False
        self.selected_square: Optional[int] = None
        self.highlighted_destinations: List[int] = []
        self.hint_sequence: Optional[List[int]] = None
        self.last_move_sequence: Optional[List[int]] = None

        # Pre-draw board grid
        self._draw_squares()

    def set_board_orientation(self, flipped: bool) -> None:
        """Set whether the board should be flipped."""
        self.flipped = flipped

    def set_selected_square(self, square: Optional[int]) -> None:
        """Set the currently selected square."""
        self.selected_square = square

    def set_highlighted_destinations(self, destinations: List[int]) -> None:
        """Set squares to highlight as possible destinations."""
        self.highlighted_destinations = destinations

    def set_hint_sequence(self, sequence: Optional[List[int]]) -> None:
        """Set the hint move sequence to display."""
        self.hint_sequence = sequence

    def set_last_move(self, move_sequence: Optional[List[int]]) -> None:
        """Set the last move sequence for highlighting."""
        self.last_move_sequence = move_sequence

    def redraw_board(self, board: Any) -> None:
        """Redraw the entire board with current state."""
        self._clear_overlays()
        self._draw_last_move_path()
        self._draw_pieces(board)
        self._draw_selection_highlight()
        self._draw_destination_highlights()
        self._draw_hint_overlay()

    def _draw_squares(self) -> None:
        """Draw the static board squares."""
        self.canvas.delete("square")
        for r in range(8):
            for c in range(8):
                x1 = c * SQUARE_SIZE
                y1 = r * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = BOARD_BG_DARK if (r + c) % 2 == 1 else BOARD_BG_LIGHT
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=color, tags=("square",)
                )

    def _clear_overlays(self) -> None:
        """Clear all overlay elements."""
        self.canvas.delete("piece")
        self.canvas.delete("sel")
        self.canvas.delete("dest")
        self.canvas.delete("last")
        self.canvas.delete("hint")

    def _draw_last_move_path(self) -> None:
        """Draw highlighting for the last move path."""
        if not self.last_move_sequence:
            return

        path = self.last_move_sequence
        for a, b in zip(path, path[1:]):
            ra, ca = rc(a)
            rb, cb = rc(b)

            if self.flipped:
                ra, ca = 7 - ra, 7 - ca
                rb, cb = 7 - rb, 7 - cb

            x1 = ca * SQUARE_SIZE
            y1 = ra * SQUARE_SIZE
            x2 = cb * SQUARE_SIZE
            y2 = rb * SQUARE_SIZE

            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=BOARD_LASTMOVE, width=4, tags=("last",)
            )

    def _draw_pieces(self, board: Any) -> None:
        """Draw all pieces on the board."""
        for idx in range(1, 33):
            piece_value = board[idx]
            if piece_value == 0:
                continue

            r, c = rc(idx)
            if self.flipped:
                r = 7 - r
                c = 7 - c

            cx = c * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = r * SQUARE_SIZE + SQUARE_SIZE // 2
            rad = int(SQUARE_SIZE * 0.36)

            fill = PIECE_BLACK_FILL if piece_value > 0 else PIECE_RED_FILL
            outline = PIECE_BLACK_OUTLINE if piece_value > 0 else PIECE_RED_OUTLINE

            self.canvas.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                fill=fill, outline=outline, width=2, tags=("piece",)
            )

            # Draw king marker
            if abs(piece_value) == 2:
                self.canvas.create_text(
                    cx, cy, text="K", fill=KING_TEXT_COLOR,
                    font=("Segoe UI", int(SQUARE_SIZE * 0.33), "bold"),
                    tags=("piece",)
                )

    def _draw_selection_highlight(self) -> None:
        """Draw highlight for the selected square."""
        if not self.selected_square:
            return

        r, c = rc(self.selected_square)
        if self.flipped:
            r = 7 - r
            c = 7 - c

        x1 = c * SQUARE_SIZE
        y1 = r * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE

        self.canvas.create_rectangle(
            x1 + 3, y1 + 3, x2 - 3, y2 - 3,
            outline=BOARD_HL_SQ, width=3, tags=("sel",)
        )

    def _draw_destination_highlights(self) -> None:
        """Draw highlights for possible destination squares."""
        for dst in self.highlighted_destinations:
            r, c = rc(dst)
            if self.flipped:
                r = 7 - r
                c = 7 - c

            cx = c * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = r * SQUARE_SIZE + SQUARE_SIZE // 2
            rad = int(SQUARE_SIZE * 0.16)

            self.canvas.create_oval(
                cx - rad, cy - rad, cx + rad, cy + rad,
                fill=BOARD_HL_DEST, outline="", tags=("dest",)
            )

    def _draw_hint_overlay(self) -> None:
        """Draw hint move path overlay."""
        if not self.hint_sequence:
            return

        for a, b in zip(self.hint_sequence, self.hint_sequence[1:]):
            ra, ca = rc(a)
            rb, cb = rc(b)

            if self.flipped:
                ra, ca = 7 - ra, 7 - ca
                rb, cb = 7 - rb, 7 - cb

            x1 = ca * SQUARE_SIZE + SQUARE_SIZE // 2
            y1 = ra * SQUARE_SIZE + SQUARE_SIZE // 2
            x2 = cb * SQUARE_SIZE + SQUARE_SIZE // 2
            y2 = rb * SQUARE_SIZE + SQUARE_SIZE // 2

            self.canvas.create_line(
                x1, y1, x2, y2, fill="#AA66FF", width=5, arrow="last", tags=("hint",)
            )
