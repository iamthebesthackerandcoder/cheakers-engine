"""
Game state management for the Checkers GUI.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import copy

from checkers.engine import initial_board, is_terminal, count_pieces
from checkers.gui.constants import SQUARE_SIZE


class GameState:
    """Manages the core game state including board, players, and history."""

    def __init__(self):
        self.board = initial_board()
        self.current_player = 1  # 1 = Black (human default), -1 = Red
        self.human_side = "Black"  # "Black" or "Red"
        self.move_number = 1
        self.last_move: Optional[List[int]] = None
        self.history: List[Tuple[Any, int, int, Optional[List[int]]]] = []
        self.game_over = False
        self.winner: Optional[str] = None

    def reset_game(self, human_side: str = "Black") -> None:
        """Reset the game to initial state."""
        self.board = initial_board()
        self.move_number = 1
        self.last_move = None
        self.history.clear()
        self.human_side = human_side
        self.current_player = 1 if human_side == "Black" else -1
        self.game_over = False
        self.winner = None

    def make_move(self, move_sequence: List[int]) -> bool:
        """Apply a move to the game state and return success."""
        if self.game_over:
            return False

        # Record history before making the move
        self.history.append((
            copy.deepcopy(self.board),
            self.current_player,
            self.move_number,
            self.last_move
        ))

        # Apply the move
        self.board = self._apply_move_sequence(move_sequence)
        self.last_move = move_sequence.copy()

        # Switch players
        self.current_player = -self.current_player
        if self.current_player == 1:
            self.move_number += 1

        # Check for game end
        self._check_game_end()
        return True

    def undo_move(self) -> bool:
        """Undo the last move and return success."""
        if not self.history:
            return False

        self.board, self.current_player, self.move_number, self.last_move = self.history.pop()
        self.game_over = False
        self.winner = None
        return True

    def _apply_move_sequence(self, move_sequence: List[int]) -> Any:
        """Apply a complete move sequence to the board."""
        from checkers.engine import apply_move
        board = self.board
        for i in range(0, len(move_sequence), 2):
            start = move_sequence[i]
            end = move_sequence[i + 1]
            board = apply_move(board, [start, end])
        return board

    def _check_game_end(self) -> None:
        """Check if the game has ended and set appropriate flags."""
        if is_terminal(self.board, self.current_player):
            self.game_over = True
            self.winner = "RED" if self.current_player == 1 else "BLACK"

    def get_current_player_name(self) -> str:
        """Get the name of the current player."""
        return "BLACK" if self.current_player == 1 else "RED"

    def get_human_player(self) -> int:
        """Get the player number for the human player."""
        return 1 if self.human_side == "Black" else -1

    def is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        return self.current_player == self.get_human_player()

    def get_piece_counts(self) -> Tuple[int, int, int, int]:
        """Get piece counts: (black_pieces, red_pieces, black_kings, red_kings)."""
        return count_pieces(self.board)

    def get_board_copy(self) -> Any:
        """Get a deep copy of the current board."""
        return copy.deepcopy(self.board)
